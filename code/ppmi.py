import json, spacy, re
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, save_npz, spdiags

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
            if target in raw_lemma and raw_lemma != target:
                splitted = raw_lemma.replace(target, ' '+target+' ')
                splitted_lemmas = [t.lemma_ for t in nlp(splitted)]
                for l in splitted_lemmas:
                    lemmas.append(l)
                break
        if not splitted_lemmas:
            lemmas.append(raw_lemma)

    return lemmas


def cooccurence_matrix(sentences, vocabulary, window_size):

    w2i = {w: i for i, w in enumerate(vocabulary)}
    matrix = dict()

    for sentence in sentences:
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


def main(output_path, targets_path, sentences_path, index2word_path, window_size, alpha, k):

    # Load targets
    with open(targets_path, "r") as infile:
        targets = [line.strip('\n') for line in infile.readlines()]
    
    # Load sentences
    with open(sentences_path, "r") as infile:
        data = [json.loads(line.strip('\n')) for line in infile.readlines()]
    sentences = [i['text'] for i in data][:10]

    # Lemmatize sentences 
    lemmatized = [lemmatize_sentence(preprocess(s), targets) for s in sentences]
    
    # Get co-occurence counts
    vocabulary = sorted(list(set([w for s in lemmatized for w in s if len(s)>1]))) # Skip one-word sentences to avoid zero-vectors
    matrix_dict = cooccurence_matrix(lemmatized, vocabulary, window_size)

    # Convert dictionary to sparse matrix
    matrix = dok_matrix((len(vocabulary),len(vocabulary)), dtype=float)
    matrix._update(matrix_dict) 
    
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

    # Eliminate negative and counts
    matrix.data[matrix.data <= 0] = 0.0
    matrix.eliminate_zeros()

    # Save matrix and index2word dict
    #with open(output_path, 'wb') as outfile:
        #save_npz(outfile, csr_matrix(matrix))  

    #i2w = {i: w for i, w in enumerate(vocabulary)}
    #with open(index2word_path, 'w') as outfile:
        #json.dump(i2w, outfile)


if __name__ == '__main__':

    targets_path = '../data/targets.txt'

    #output_path = '../output/ppmi_Forum_Democratie1.npz'
    #sentences_path = '../data/subreddit_Forum_Democratie_comments_partial1.jsonl'
    #contexts_path = '../output/ppmicontexts_Forum_Democratie1.json'

    output_path = '../output/ppmi_Poldersocialisme.npz'
    sentences_path = '../data/subreddit_Poldersocialisme_comments.jsonl'
    index2word_path = '../output/ppmi_index2word_Poldersocialisme.json'

    window_size = 10
    alpha = 0.75
    k = 5

    main(output_path, targets_path, sentences_path, index2word_path, window_size, alpha, k)

