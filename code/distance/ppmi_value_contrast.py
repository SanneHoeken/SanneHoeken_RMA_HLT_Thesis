import numpy as np
import json, argparse
from scipy.sparse import load_npz

def main(targets_path, ppmi_path1, ppmi_path2, vocab_path1, vocab_path2, 
            counter_path1, counter_path2, k, thres):
    """
    Compute cosine distance between ppmi vectors
    """

    # Load targets
    with open(targets_path, "r") as infile:
        targets = [line.strip('\n') for line in infile.readlines()]
    # Load vocabularies
    with open(vocab_path1, "r") as infile:
        vocab1 = [line.strip('\n') for line in infile.readlines()]
    w2i1 = {w: i for i, w in enumerate(vocab1)}
    
    with open(vocab_path2, "r") as infile:
        vocab2 = [line.strip('\n') for line in infile.readlines()]
    w2i2 = {w: i for i, w in enumerate(vocab2)}

    # Load counters
    with open(counter_path1, "r") as infile:
        counter1 = json.load(infile)
    for word in w2i1:
        if word not in counter1:
            counter1[word] = 0
    with open(counter_path2, "r") as infile:
        counter2 = json.load(infile)
    for word in w2i2:
        if word not in counter2:
            counter2[word] = 0

    # Get ppmi matrices
    ppmi1 = load_npz(ppmi_path1)
    ppmi2 = load_npz(ppmi_path2)

    # Get vocab intersection and intersected column ids
    vocab_intersect = sorted(list(set(vocab1).intersection(vocab2)))
    intersected_columns1 = [int(w2i1[item]) for item in vocab_intersect]
    intersected_columns2 = [int(w2i2[item]) for item in vocab_intersect]

    for target in targets:
        print('\n'+target+'\n')
        vector1 = ppmi1[:, w2i1[target]].toarray().flatten()[intersected_columns1]
        vector2 = ppmi2[:, w2i2[target]].toarray().flatten()[intersected_columns2]
        diff1 = vector1 - vector2
        diff2 = vector2 - vector1
        sorted_idx1 = np.flip(np.argsort(diff1).tolist())
        sorted_idx2 = np.flip(np.argsort(diff2).tolist())
        
        i = 0
        print(f"{k} most distinguishing words of dataset1 w.r.t dataset2:")
        words = []
        for idx in sorted_idx1:
            word = vocab_intersect[idx]
            if counter1[word] >= thres and word.isalpha():
                words.append(word)
                i+=1
                if i == k:
                    print(words)
                    break
        
        i = 0
        print(f"{k} most distinguishing words of dataset2 w.r.t dataset1:")
        words = []
        for idx in sorted_idx2:
            word = vocab_intersect[idx]
            if counter2[word] >= thres and word.isalpha():
                words.append(word)
                i+=1
                if i == k:
                    print(words)
                    break


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets_path",
                        help="Path to txt-file with target words",
                        type=str)
    parser.add_argument("--ppmi_path1",
                        help="Path to file with ppmi representations of target words in first dataset in npz-format",
                        type=str)
    parser.add_argument("--ppmi_path2",
                        help="Path to file with ppmi representations of target words in second dataset in npz-format",
                        type=str)
    parser.add_argument("--vocab_path1",
                        help="Path to txt-file with vocabulary of first dataset",
                        type=str)
    parser.add_argument("--vocab_path2",
                        help="Path to txt-file with vocabulary of second dataset",
                        type=str)
    parser.add_argument("--counter_path1",
                        help="Path to json-file with vocabulary words of first dataset mapping to their frequency",
                        type=str)
    parser.add_argument("--counter_path2",
                        help="Path to json-file with vocabulary words of second dataset mapping to their frequency",
                        type=str)
    parser.add_argument("--k",
                        help="Number of distinguishing words",
                        default=10,
                        type=int)
    parser.add_argument("--thres",
                        help="Frequency threshold for each word to be included",
                        default=10,
                        type=int)
    args = parser.parse_args()
    main(args.targets_path, args.ppmi_path1, args.ppmi_path2, args.vocab_path1, args.vocab_path2,
        args.counter_path1, args.counter_path2, args.k, args.thres)