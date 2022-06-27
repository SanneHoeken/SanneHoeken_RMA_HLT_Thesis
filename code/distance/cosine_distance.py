from scipy.sparse import load_npz
from scipy.spatial.distance import cosine
from tqdm import tqdm
import argparse

def main(targets_path, ppmi_path1, ppmi_path2, vocab_path1, vocab_path2, output_path):
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

    # Get ppmi matrices
    ppmi1 = load_npz(ppmi_path1)
    ppmi2 = load_npz(ppmi_path2)

    # Get vocab intersection and intersected column ids
    vocab_intersect = sorted(list(set(vocab1).intersection(vocab2)))
    intersected_columns1 = [int(w2i1[item]) for item in vocab_intersect]
    intersected_columns2 = [int(w2i2[item]) for item in vocab_intersect]

    # Print cosine distance of targets to output file
    with open(output_path, 'w') as outfile:
        for target in tqdm(targets):
            vector1 = ppmi1[:, w2i1[target]].toarray().flatten()[intersected_columns1]
            vector2 = ppmi2[:, w2i2[target]].toarray().flatten()[intersected_columns2]
            distance = cosine(vector1, vector2) 
            outfile.write(f'{target}\t{distance}\n')


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
    parser.add_argument("--output_path",
                        help="Path to csv-file to write the results to",
                        type=str)
    args = parser.parse_args()
    main(args.targets_path, args.ppmi_path1, args.ppmi_path2, args.vocab_path1, args.vocab_path2, args.output_path)