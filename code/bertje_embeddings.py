import torch, json, spacy
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SequentialSampler

nlp = spacy.load("nl_core_news_sm", disable=['parser', 'ner'])

def encode_sentence(sentence, targets, tokenizer):

    # lemmatize and split compounds with targetword in it 
    lemmas = [t.lemma_ for t in nlp(sentence)]
    new_lemmas = []
    for lemma in lemmas:
        for target in targets:
            if target in lemma:
                lemma = lemma.replace(target, ' '+target+' ')
                break
        new_lemmas.append(lemma)
    lemmas = " ".join(new_lemmas)
    encoded = tokenizer.encode(lemmas, add_special_tokens=False)

    return encoded


def get_encoded_context(token_ids, target_position, sequence_length):
    
    # -2 as [CLS] and [SEP] tokens will be added later; /2 as it's a one-sided window
    window_size = int((sequence_length - 2) / 2)
    context_start = max([0, target_position - window_size])
    padding_offset = max([0, window_size - target_position])
    padding_offset += max([0, target_position + window_size - len(token_ids)])

    context_ids = token_ids[context_start:target_position + window_size]
    context_ids += padding_offset * [0]

    new_target_position = target_position - context_start

    return context_ids, new_target_position


class ContextsDataset(Dataset):

    def __init__(self, targets, sentences, tokenizer):
        super(ContextsDataset).__init__()
        self.data = []
        self.textdata = dict()
        self.context_length = tokenizer.model_max_length    
        self.window_size = 10
        
        # Store vocabulary indices of target words and itialize counter
        targets_ids = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
        i2w = {t_id[0]: t for t, t_id in zip(targets, targets_ids)}
        self.target_counter = {i2w[target]: 0 for target in i2w}

        CLS_id = tokenizer.encode('[CLS]', add_special_tokens=False)[0]
        SEP_id = tokenizer.encode('[SEP]', add_special_tokens=False)[0]

        for sentence in tqdm(sentences, total=len(sentences)):
            token_ids = encode_sentence(sentence, targets, tokenizer)
            for spos, tok_id in enumerate(token_ids):
                if tok_id in i2w:
                    # update counter
                    self.target_counter[i2w[tok_id]] += 1

                    # get encoded context
                    context_ids, pos_in_context = get_encoded_context(token_ids, spos, self.context_length)
                    input_ids = [CLS_id] + context_ids + [SEP_id]
                    self.data.append((input_ids, i2w[tok_id], pos_in_context))

                    # get textual context window per target
                    words = sentence.split()
                    lower = max(spos - self.window_size, 0)
                    upper = min(spos + self.window_size, len(words))
                    window = words[lower:spos] + words[spos+1:upper+1]
                    if i2w[tok_id] not in self.textdata:
                        self.textdata[i2w[tok_id]] = []
                    self.textdata[i2w[tok_id]].append(window)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, lemma, pos_in_context = self.data[index]
        return torch.tensor(input_ids), lemma, pos_in_context


def set_seed(seed, n_gpus):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus > 0:
        torch.cuda.manual_seed_all(seed)


def main(model_dir, output_path, targets_path, sentences_path, contexts_path, batch_size):

    # Setup CUDA / GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    set_seed(42, n_gpu)

    # Load targets
    with open(targets_path, "r") as infile:
        targets = [line.strip('\n') for line in infile.readlines()]
    
    # Load sentences
    with open(sentences_path, "r") as infile:
        data = [json.loads(line.strip('\n')) for line in infile.readlines()]
    sentences = [i['text'] for i in data]
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, never_split=targets)
    model = AutoModel.from_pretrained(model_dir, output_hidden_states=True)
    model.to(device)

    # Dataset with all contexts of target words
    dataset = ContextsDataset(targets, sentences, tokenizer)
    
    # Store contexts per target word in json
    with open(contexts_path, "w") as outfile:
        json.dump(dataset.textdata, outfile)

    # Containers for usage representations
    nDims = model.config.hidden_size
    nLayers = model.config.num_hidden_layers
    
    usages = {target: np.empty((target_count, (nLayers + 1) * nDims))  
        for (target, target_count) in dataset.target_counter.items()}
    curr_idx = {target: 0 for target in dataset.target_counter}
    
    # Iterate over sentences and collect representations
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        batch_tuple = tuple()
        for t in batch:
            try:
                batch_tuple += (t.to(device),)
            except AttributeError:
                batch_tuple += (t,)

        batch_input_ids = batch_tuple[0].squeeze(1)
        batch_lemmas, batch_spos = batch_tuple[1], batch_tuple[2]

        # get hidden layer representations for batch
        with torch.no_grad():
            if torch.cuda.is_available():
                batch_input_ids = batch_input_ids.to('cuda')

            outputs = model(batch_input_ids)

            if torch.cuda.is_available():
                hidden_states = [l.detach().cpu().clone().numpy() for l in outputs[2]]
            else:
                hidden_states = [l.clone().numpy() for l in outputs[2]]

            # store all usage representations (hidden layers) per target word in dictionary
            for b_id in np.arange(len(batch_input_ids)):
                lemma = batch_lemmas[b_id]
                layers = [layer[b_id, batch_spos[b_id] + 1, :] for layer in hidden_states]
                usage_vector = np.concatenate(layers)
                usages[lemma][curr_idx[lemma], :] = usage_vector
                curr_idx[lemma] += 1

    np.savez_compressed(output_path, **usages)
    print(curr_idx)
    
    
if __name__ == '__main__':

    targets_path = '../data/targets.txt'
    model_dir = '../models/bertje-ft-all'
    batch_size = 8

    for community, com in [('Poldersocialisme', 'PS'), 
                            ('Forum_Democratie1', 'FD1'), 
                            ('ForumDemocratie2', 'FD2')]:
        
        output_path = f'../output/embeddings/bertje-ft-all_embeddings_{com}.npz'
        sentences_path = f'../data/subreddit_{community}_comments.jsonl'
        contexts_path = f'../output/contexts/contexts_{com}.json'
        
        print('\n', community, '\n')
        main(model_dir, output_path, targets_path, sentences_path, contexts_path, batch_size)