import torch, json, spacy
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler

nlp = spacy.load("nl_core_news_sm", disable=['parser', 'ner'])

def mask_tokens(inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling:
    80% MASK, 10% random, 10% original. """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training with probability 0.15 
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


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


class RedditDataset(Dataset):
    
    def __init__(self, sentences, targets, tokenizer):
        self.examples = [encode_sentence(sentence, targets, tokenizer) for sentence in sentences]
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def set_seed(seed, n_gpus):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus > 0:
        torch.cuda.manual_seed_all(seed)


def main(model_dir, output_dir, targets_path, sentences_path, batch_size, epochs):

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
    model = AutoModelForMaskedLM.from_pretrained(model_dir)
    model.to(device)

    add_unknown_targets(model, tokenizer, targets)

    # Create dataset and set scheduler for training
    dataset = RedditDataset(sentences, targets, tokenizer)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.zero_grad()
    for i in range(epochs):
        print(f"Epoch {i+1} out of {epochs}...")
        for step, batch in enumerate(tqdm(dataloader)):
            inputs, labels = mask_tokens(batch, tokenizer)
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()
            outputs = model(inputs, labels=labels) 
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step() 
            model.zero_grad()
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    

if __name__ == '__main__':
    
    model_dir = "GroNLP/bert-base-dutch-cased"
    targets_path = '../data/targets.txt'
    output_dir = "../models/bertje-ft-all"
    sentences_path = '../data/subreddit_all_comments.jsonl'
    batch_size = 8
    epochs = 3

    main(model_dir, output_dir, targets_path, sentences_path, batch_size, epochs)