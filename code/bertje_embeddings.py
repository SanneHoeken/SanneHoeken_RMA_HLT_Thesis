from collections import defaultdict
import torch, json, spacy, os.path
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

nlp = spacy.load("nl_core_news_sm", disable=['parser', 'ner'])

def id_to_target(model, tokenizer, targets):
    
    unk_id = tokenizer.convert_tokens_to_ids('[UNK]')
    targets_ids = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
    i2w = {}
    for t, t_id in zip(targets, targets_ids):
        if len(t_id) > 1 or (len(t_id) == 1 and t_id == unk_id):
            if tokenizer.add_tokens([t]):
                print(f"'{t}' is added to model's vocabulary")
                model.resize_token_embeddings(len(tokenizer))
                i2w[len(tokenizer) - 1] = t
        else:
            i2w[t_id[0]] = t
    
    return i2w


def count_targets(i2w, tokenizer, sentences):
    
    target_counter = {i2w[target]: 0 for target in i2w}
    for sentence in tqdm(sentences):
        # lemmatize and split compounds with targetword in it 
        lemmas = [t.lemma_ for t in nlp(sentence)]
        new_lemmas = []
        for lemma in lemmas:
            for target in target_counter:
                if target in lemma:
                    lemma = lemma.replace(target, ' '+target+' ')
                    break
            new_lemmas.append(lemma)
        lemmas = " ".join(new_lemmas)
        encoded = tokenizer.encode(lemmas, add_special_tokens=False)
        
        for tok_id in encoded:
            if tok_id in i2w:
                target_counter[i2w[tok_id]] += 1
        
    return target_counter


def get_context(token_ids, target_position, sequence_length):
    
    # -2 as [CLS] and [SEP] tokens will be added later; /2 as it's a one-sided window
    window_size = int((sequence_length - 2) / 2)
    context_start = max([0, target_position - window_size])
    padding_offset = max([0, window_size - target_position])
    padding_offset += max([0, target_position + window_size - len(token_ids)])

    context_ids = token_ids[context_start:target_position + window_size]
    context_ids += padding_offset * [0]

    new_target_position = target_position - context_start

    return context_ids, new_target_position


class ContextsDataset(torch.utils.data.Dataset):

    def __init__(self, targets_i2w, sentences, tokenizer, sequence_length):
        super(ContextsDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.CLS_id = tokenizer.encode('[CLS]', add_special_tokens=False)[0]
        self.SEP_id = tokenizer.encode('[SEP]', add_special_tokens=False)[0]

        for sentence in tqdm(sentences, total=len(sentences)):
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            for spos, tok_id in enumerate(token_ids):
                if tok_id in targets_i2w:
                    context_ids, pos_in_context = get_context(token_ids, spos, sequence_length)
                    input_ids = [self.CLS_id] + context_ids + [self.SEP_id]
                    self.data.append((input_ids, targets_i2w[tok_id], pos_in_context))

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


def main(model_dir, output_path, targets_path, sentences_path, counter_path, batch_size, context_length):

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

    # Store vocabulary indices of target words
    i2w = id_to_target(model, tokenizer, targets)
    
    # Count usages of target words
    if os.path.exists(counter_path):
        with open(counter_path, "r") as infile:
            target_counter = json.load(infile)
    else:
        target_counter = count_targets(i2w, tokenizer, sentences)
        with open(counter_path, "w") as outfile:
            json.dump(target_counter, outfile)
    
    # Containers for usages
    nDims = model.config.hidden_size
    nLayers = model.config.num_hidden_layers
    
    usages = {target: np.empty((target_count, (nLayers + 1) * nDims))  
        for (target, target_count) in target_counter.items()}
    curr_idx = {target: 0 for target in target_counter}
    
    # Dataset with all contexts of target words
    dataset = ContextsDataset(i2w, sentences, tokenizer, context_length)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    iterator = tqdm(dataloader, desc="Iteration")
    
    # Iterate over sentences and collect representations
    for step, batch in enumerate(iterator):
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
    model_dir = "GroNLP/bert-base-dutch-cased"
    output_path = '../output/bertje_embeddings_Forum_Democratie.npz'
    sentences_path = '../data/subreddit_Forum_Democratie_comments.jsonl'
    targets_path = '../data/targets.txt'
    counter_path = '../output/target_counts_Forum_Democratie.json'

    batch_size = 8
    context_length = 512

    main(model_dir, output_path, targets_path, sentences_path, counter_path, batch_size, context_length)