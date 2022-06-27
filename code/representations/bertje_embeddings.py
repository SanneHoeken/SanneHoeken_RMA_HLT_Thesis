import torch, json, spacy, argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SequentialSampler

nlp = spacy.load("nl_core_news_sm", disable=['parser', 'ner'])

def add_unknown_targets(model, tokenizer, targets):

    unk_id = tokenizer.convert_tokens_to_ids('[UNK]')
    targets_ids = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
    for t, t_id in zip(targets, targets_ids):
        if len(t_id) > 1 or (len(t_id) == 1 and t_id == unk_id):
            if tokenizer.add_tokens([t]):
                print(f"'{t}' is added to model's vocabulary")
                model.resize_token_embeddings(len(tokenizer))


def encode_sentence(sentence, targets, tokenizer, lemma=False):
    """
    Tokenize or lemmatize sentence
    Split compounds with targetword in it 
    Encode sentence
    """
    if lemma:
        wordforms = [t.lemma_ for t in nlp(sentence)]
    else:
        wordforms = [t.text for t in nlp(sentence)]

    new_wordforms = []
    for wordform in wordforms:
        for target in targets:
            if target in wordform:
                wordform = wordform.replace(target, ' '+target+' ')
                break
        new_wordforms.append(wordform)
    wordforms = " ".join(new_wordforms)
    encoded = tokenizer.encode(wordforms, 
                                max_length=tokenizer.model_max_length, 
                                padding='max_length',
                                truncation=True)

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

    def __init__(self, targets, sentences, tokenizer, lemma):
        super(ContextsDataset).__init__()
        self.data = []
        self.context_length = tokenizer.model_max_length    

        
        # Store vocabulary indices of target words and itialize counter
        targets_ids = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
        i2w = {t_id[0]: t for t, t_id in zip(targets, targets_ids)}
        self.target_counter = {i2w[target]: 0 for target in i2w}

        CLS_id = tokenizer.encode('[CLS]', add_special_tokens=False)[0]
        SEP_id = tokenizer.encode('[SEP]', add_special_tokens=False)[0]

        for sentence in tqdm(sentences, total=len(sentences)):
            token_ids = encode_sentence(sentence, targets, tokenizer, lemma)
            for spos, tok_id in enumerate(token_ids):
                if tok_id in i2w:
                    # update counter
                    self.target_counter[i2w[tok_id]] += 1

                    # get encoded context
                    context_ids, pos_in_context = get_encoded_context(token_ids, spos, self.context_length)
                    input_ids = [CLS_id] + context_ids + [SEP_id]
                    self.data.append((input_ids, i2w[tok_id], pos_in_context))


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


def main(model_dir, output_path, targets_path, sentences_path, batch_size, pretrained, lemma, layers):

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

    if pretrained:
        add_unknown_targets(model, tokenizer, targets)

    # Dataset with all contexts of target words
    print('Generating context dataset...')
    dataset = ContextsDataset(targets, sentences, tokenizer, lemma)
    
    # Containers for usage representations
    nDims = model.config.hidden_size
    if layers == 'toplayer':
        nLayers = 1
    else:
        nLayers = model.config.num_hidden_layers + 1
    usages = {target: np.empty((target_count, nLayers * nDims))  
        for (target, target_count) in dataset.target_counter.items()}
    curr_idx = {target: 0 for target in dataset.target_counter}
    
    # Iterate over sentences and collect representations
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    print('Extract usage representations...')
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
                if layers == 'toplayer':
                    usage_vector = layers[-1]
                else:
                    usage_vector = np.concatenate(layers)
                usages[lemma][curr_idx[lemma], :] = usage_vector
                curr_idx[lemma] += 1

    np.savez_compressed(output_path, **usages)
    print(curr_idx)
    
    
if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",
                        help="""Path to data file. This file must be in jsonl-format 
                        with json-objects containing the member 'text'""",
                        type=str)
    parser.add_argument("--targets_path",
                        help="Path to txt-file with target words",
                        type=str)
    parser.add_argument("--model_dir",
                        help="Directory of BERT-based model available via Hugging Face's transformer library",
                        type=str)
    parser.add_argument("--output_path",
                        help="Path to npz-file in which usage representations of target words should be stored",
                        type=str)
    parser.add_argument("--batch_size",
                        default=8,
                        type=int)
    parser.add_argument("--pretrained",
                        help="whether the model is pretrained (True) or finetuned (False)",
                        choices=[True, False],
                        default=False,
                        type=bool)
    parser.add_argument("--lemma",
                        help="whether the data should be lemmatized (True) or not (False)",
                        choices=[True, False],
                        default=False,
                        type=bool)
    parser.add_argument("--layers",
                        help="the selection of hidden layers to extract as usage representation",
                        choices=['all', 'toplayer'],
                        default='all',
                        type=str)
    args = parser.parse_args()
    main(args.model_dir, args.output_path, args.targets_path, args.dataset_path, args.batch_size, args.pretrained, args.lemma, args.layers)
