import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

# Average pairwise distance (APD) algorithm

def mean_pairwise_distance(usage_matrix1, usage_matrix2, metric):
    """
    Computes the mean pairwise distance between two usage matrices.
    """
    if usage_matrix1.shape[0] == 0 or usage_matrix2.shape[0] == 0:
        return 0.

    return np.mean(cdist(usage_matrix1, usage_matrix2, metric=metric))


def main(targets_path, usage_path1, usage_path2, output_path, distmetric):
    """
    Compute (diachronic) distance between sets of contextualised representations.
    The distance metric must be compatible with `scipy.spatial.distance.cdist`
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
            distance = mean_pairwise_distance(usages1[target], usages2[target], distmetric) 
            outfile.write(f'{target}\t{distance}\n')


if __name__ == '__main__':
    
    targets_path = '../data/targets.txt'
    usage_path1 = '../output/embeddings/bertje-ft-all_embeddings_FD1.npz'
    usage_path2 = '../output/embeddings/bertje-ft-all_embeddings_PS.npz'
    output_path = '../output/apd/apd_FD1-ft-all_PS-ft-all.csv'
    distmetric = 'cosine'
    
    main(targets_path, usage_path1, usage_path2, output_path, distmetric)

    