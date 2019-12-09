import numpy as np
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('-l', '--losses', nargs='+', required=True,
    help='The losses from the K-Fold Cross Validation experiment')
argparser.add_argument('-a', '--accs', nargs='+', required=True,
    help='The accuracies from the K-Fold Cross Validation experiment')

def calculate_stats(nums):
    """ This function accepts losses and accuracies and calculates
    the mean, standard deviation, and variance for each. The results
    are printed to standard output."""

    nums = np.array(nums).astype(np.float)

    mean = np.mean(nums)
    std = np.std(nums)
    var = np.var(nums)

    msg = f'\tMean: {mean}\n\tStandard Deviation: {std}\n\tVariance: {var}'
    print(msg)

def main():
    args = argparser.parse_args()
    losses = args.losses
    accs = args.accs
    
    print('Losses:')
    calculate_stats(losses)

    print('Accuracies:')
    calculate_stats(accs)

if __name__ == "__main__":
    main()
