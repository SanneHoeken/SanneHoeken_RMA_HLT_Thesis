import json, spacy, re, argparse
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, spdiags, save_npz
from tqdm import tqdm
from collections import Counter

nlp = spacy.load("nl_core_news_sm", disable=['parser', 'ner'])

def preprocess(text):
    
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    text = text.replace('  ', ' ').replace('  ', ' ').rstrip().lstrip()
    text = text.lower()

    return text


def lemmatize_sentence(sentence, targets):

    # lemmatize and split compounds with targetword in it 
    raw_lemmas = [t.lemma_ for t in nlp(sentence)]
    lemmas = []
    for raw_lemma in raw_lemmas:
        splitted_lemmas = None
        for target in targets:
            if target in raw_lemma:
                if raw_lemma != target:
                    splitted = raw_lemma.replace(target, ' '+target+' ')
                    splitted_lemmas = [t.lemma_ for t in nlp(splitted)]
                    for l in splitted_lemmas:
                        lemmas.append(l)
                break
        if not splitted_lemmas:
            lemmas.append(raw_lemma)

    return lemmas


def cooccurence_matrix(sentences, w2i, window_size):

    matrix = dict()

    for sentence in tqdm(sentences):
        for i, word in enumerate(sentence):
            lowerWindowSize = max(i-window_size, 0)
            upperWindowSize = min(i+window_size, len(sentence))
            window = sentence[lowerWindowSize:i] + sentence[i+1:upperWindowSize+1]
            if len(window)==0: # Skip one-word sentences
                continue
            windex = w2i[word]
            for contextWord in window:
                if (windex,w2i[contextWord]) not in matrix:
                    matrix[(windex,w2i[contextWord])] = 0
                matrix[(windex, w2i[contextWord])] += 1

    return matrix


def main(output_path, targets_path, sentences_path, window_size, alpha, k):
    
    # Load targets
    with open(targets_path, "r") as infile:
        targets = [line.strip('\n') for line in infile.readlines()]
    
    # Load sentences
    with open(sentences_path, "r") as infile:
        data = [json.loads(line.strip('\n')) for line in infile.readlines()]
    sentences = [i['text'] for i in data]
    
    # Lemmatize sentences 
    print('Preprocessing sentences...')
    lemmatized = [lemmatize_sentence(preprocess(s), targets) for s in tqdm(sentences)]
    
    # initialize vocabulary and save index2word dict
    print('Initializing vocabulary...')
    all_lemmas = [w for s in lemmatized for w in s]
    counter = Counter(all_lemmas)

    with open(output_path+'counter.json', 'w') as outfile:
        json.dump(counter, outfile)
    
    vocabulary = sorted(list(set(all_lemmas)))
    with open(output_path+'vocab.txt', 'w') as outfile:
        for word in vocabulary:
            outfile.write(word+'\n')
    w2i = {w: i for i, w in enumerate(vocabulary)}
    
    
    # Get co-occurence counts of whole corpus
    print('Get co-occurence matrix...')
    matrix_dict = cooccurence_matrix(lemmatized, w2i, window_size)

    # Convert dictionary to sparse matrix
    matrix = dok_matrix((len(vocabulary),len(vocabulary)), dtype=float)
    matrix._update(matrix_dict) 
    
    print('Compute ppmi matrix...')
    # Get probabilities
    row_sum = matrix.sum(axis = 1)
    col_sum = matrix.sum(axis = 0)

    # Compute smoothed P_alpha(c)
    smooth_col_sum = np.power(col_sum, alpha)
    col_sum = smooth_col_sum/smooth_col_sum.sum()

    # Compute P(w)
    row_sum = row_sum.astype(np.double)
    row_sum[row_sum != 0] = np.array(1.0/row_sum[row_sum != 0]).flatten()
    col_sum = col_sum.astype(np.double)
    col_sum[col_sum != 0] = np.array(1.0/col_sum[col_sum != 0]).flatten()

    # Apply epmi weighting (without log)
    diag_matrix = spdiags(row_sum.flatten(), [0], row_sum.flatten().size, row_sum.flatten().size, format = 'csr')
    matrix = csr_matrix(diag_matrix * matrix)
    
    diag_matrix = spdiags(col_sum.flatten(), [0], col_sum.flatten().size, col_sum.flatten().size, format = 'csr')
    matrix = csr_matrix(matrix * diag_matrix)

    # Apply log weighting and shift values
    matrix.data = np.log(matrix.data)
    matrix.data -= np.log(k)

    # Eliminate negative and zero counts
    matrix.data[matrix.data <= 0] = 0.0
    matrix.eliminate_zeros()

    # Save ppmi matrix
    print('Save ppmi matrix...')
    save_npz(output_path+'vocab.npz', csr_matrix(matrix))

    # Extract and save target vectors
    print('Save usage vectors...')
    usages = {target: matrix[:, w2i[target]].toarray().flatten().tolist() for target in targets}
    #np.savez_compressed(output_path, **usages)
    with open(output_path, 'w') as outfile:
        json.dump(usages, outfile)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",
                        help="""Path to data file. This file must be in jsonl-format 
                        with json-objects containing the member 'text'""",
                        type=str)
    parser.add_argument("--targets_path",
                        help="Path to txt-file with target words",
                        type=str)
    parser.add_argument("--output_dir",
                        help="Directory path to write results to",
                        type=str)
    parser.add_argument("--window_size",
                        default=10,
                        type=int)
    parser.add_argument("--alpha",
                        default=0.75,
                        type=float)
    parser.add_argument("--k",
                        default=5,
                        type=int)
    args = parser.parse_args()

    main(args.output_dir, args.targets_path, args.dataset_path, args.window_size, args.alpha, args.k)
