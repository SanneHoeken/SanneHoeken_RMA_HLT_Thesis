import json, argparse
from scipy.sparse import load_npz
from scipy.spatial.distance import cosine
import numpy as np
from tqdm import tqdm

def main(targets_path, ppmi_path1, ppmi_path2, vocab_path1, vocab_path2, 
            counter_path1, counter_path2, output_path, k, thres):

    # Load targets
    with open(targets_path, "r") as infile:
        targets = [line.strip('\n') for line in infile.readlines()]

    # Load vocabularies
    with open(vocab_path1, "r") as infile:
        vocab1 = [line.strip('\n') for line in infile.readlines()]
    w2i1 = {w: i for i, w in enumerate(vocab1)}
    i2w1 = {i:w for w, i in w2i1.items()}
    
    with open(vocab_path2, "r") as infile:
        vocab2 = [line.strip('\n') for line in infile.readlines()]
    w2i2 = {w: i for i, w in enumerate(vocab2)}
    i2w2 = {i:w for w, i in w2i2.items()}

    # Load counters
    with open(counter_path1, "r") as infile:
        counter1 = json.load(infile)
    with open(counter_path2, "r") as infile:
        counter2 = json.load(infile)
    

    # Get ppmi matrices
    ppmi1 = load_npz(ppmi_path1)
    ppmi2 = load_npz(ppmi_path2)

    with open(output_path, 'w') as outfile:
        outfile.write('Target\tCommunity\tPos\tCosDist\tFreq\tLemma\tPos_in_other_community\n')
        for target in targets:
            print(f'Compute cosine distances between "{target}" and all tokens in dataset 1...')
            target_vec1 = ppmi1[:, w2i1[target]].toarray().flatten()
            target_cosdists1 = np.array([cosine(target_vec1, x.toarray().flatten()) for x in tqdm(ppmi1)])
            sorted_idx1 = np.argsort(target_cosdists1).tolist()
            filtered_sorted_idx1 = [idx for idx in sorted_idx1 if i2w1[idx].isalpha() and int(counter1[i2w1[idx]]) >= thres]
            print(f'Compute cosine distances between "{target}" and all tokens in dataset 2...')
            target_vec2 = ppmi2[:, w2i2[target]].toarray().flatten()
            target_cosdists2 = np.array([cosine(target_vec2, x.toarray().flatten()) for x in tqdm(ppmi2)])
            sorted_idx2 = np.argsort(target_cosdists2).tolist()
            filtered_sorted_idx2 = [idx for idx in sorted_idx2 if i2w2[idx].isalpha() and int(counter2[i2w2[idx]]) >= thres]
        
            print(f'Write {k} nearest neighbors of "{target}" in both communities to file...')
            for i, idx in enumerate(filtered_sorted_idx1):
                word = i2w1[idx]
                freq = int(counter1[word])
                if word != target:   
                    cos_dist = round((1-target_cosdists1[idx]), 2)
                    if word not in w2i2:
                        pos_in_other = 0
                    elif w2i2[word] not in filtered_sorted_idx2:
                        pos_in_other = 0
                    else:
                        pos_in_other = int(filtered_sorted_idx2.index(w2i2[word]))
                    outfile.write(f'{target}\tdataset1\t{i}\t{cos_dist}\t{freq}\t{word}\t{pos_in_other}\n')
                if i > k:
                    break
            
            for i, idx in enumerate(filtered_sorted_idx2):
                word = i2w2[idx]
                freq = int(counter2[word])
                if word != target:   
                    cos_dist = round((1-target_cosdists2[idx]), 2)
                    if word not in w2i1:
                        pos_in_other = 0
                    elif w2i1[word] not in filtered_sorted_idx1:
                        pos_in_other = 0
                    else:
                        pos_in_other = int(filtered_sorted_idx1.index(w2i1[word]))
                    outfile.write(f'{target}\tdataset2\t{i}\t{cos_dist}\t{freq}\t{word}\t{pos_in_other}\n')
                if i > k:
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
    parser.add_argument("--output_path",
                        help="Path to csv-file to write the results to",
                        type=str)
    parser.add_argument("--k",
                        help="Number of nearest neighbors",
                        default=10,
                        type=int)
    parser.add_argument("--thres",
                        help="Frequency threshold for each word to be included",
                        default=10,
                        type=int)
    args = parser.parse_args()
    main(args.targets_path, args.ppmi_path1, args.ppmi_path2, args.vocab_path1, args.vocab_path2,
        args.counter_path1, args.counter_path2, args.output_path, args.k, args.thres)
