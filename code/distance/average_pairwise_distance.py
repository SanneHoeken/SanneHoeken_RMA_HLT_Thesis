import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import argparse

# Average pairwise distance (APD) algorithm

def mean_pairwise_distance(usage_matrix1, usage_matrix2):
    """
    Computes the mean pairwise distance between two usage matrices.
    """
    if usage_matrix1.shape[0] == 0 or usage_matrix2.shape[0] == 0:
        return 0.

    return 1 - np.mean(cdist(usage_matrix1, usage_matrix2, metric='cosine'))


def main(targets_path, usage_path1, usage_path2, output_path):
    """
    Compute cosine distance between sets of contextualised representations.
    """

    # Load targets
    with open(targets_path, "r") as infile:
        targets = [line.strip('\n') for line in infile.readlines()]
    
    # Get usages collected from corpus 1 and 2
    usages1 = np.load(usage_path1)
    usages2 = np.load(usage_path2)

    # Print only targets to output file
    with open(output_path, 'w') as outfile:
        for target in tqdm(targets):
            distance = mean_pairwise_distance(usages1[target], usages2[target]) 
            outfile.write(f'{target}\t{distance}\n')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets_path",
                        help="Path to txt-file with target words",
                        type=str)
    parser.add_argument("--usage_path1",
                        help="Path to file with usage representations of target words in first dataset in npz-format",
                        type=str)
    parser.add_argument("--usage_path2",
                        help="Path to file with usage representations of target words in second dataset in npz-format",
                        type=str)
    parser.add_argument("--output_path",
                        help="Path to csv-file to write the results to",
                        type=str)
    args = parser.parse_args()
    main(args.targets_path, args.usage_path1, args.usage_path2, args.output_path)

    