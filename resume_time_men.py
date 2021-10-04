import argparse
import numpy as np

from time_mem.mprof import read_mprofile_file

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=["polarity", "20ng", "webkb", "r8"],
        help='dataset name',
        required=True
    )   
    parser.add_argument('--window', type=int, help='window size', required=True)
    return parser.parse_args()

def print_results(values, strategy, cut=0):
    memory = np.asarray(values.get('mem_usage')).max()
    t = np.asarray(values.get('timestamp'))
    global_start = float(t[0])
    t = t - global_start
    time = t[-1]
    print(f'METRIC: {strategy}, CUT: {cut}')
    print(f'MAX MEMORY: {memory}')
    print(f'TIME: {time}')



def main():

    args = read_args()
    dataset = args.dataset
    window = args.window


    print('===== BASELINE RESULTS =====')
    filename = (
        f'time_mem/mprofile.{dataset}.no_weight.{window}.0.dat'
    )
    values = read_mprofile_file(filename=filename)
    print_results(values, 'no_weight')


    strategies = ["chi_square", "chi_square_all", "llr", "llr_all", "pmi", "pmi_all"]
    cuts = [5, 10, 20, 30, 50, 70, 80, 90]

    print('===== PROPOSAL RESULTS =====')
    for strategy in strategies:
        print("\n\n")
        for cut in cuts:
            filename = (
                f'time_mem/mprofile.{dataset}.{strategy}.{window}.{cut}.dat'
            )
            values = read_mprofile_file(filename=filename)
            print_results(values, strategy, cut)

if __name__ == "__main__":
    main()
