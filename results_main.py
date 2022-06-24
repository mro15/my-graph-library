#! /usr/bin/env python3

import argparse
import os
import numpy as np
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats import ttest_ind

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=["polarity", "20ng", "webkb", "r8"],
        help='dataset name',
        required=True
    )   
    parser.add_argument('--emb_dim', type=int, help='embeddings dimension', required=True)
    parser.add_argument('--window', type=int, help='window size', required=True)
    parser.add_argument(
        '--cut_percent',
        type=int,
        help='percentage of edges to cut',
        required=True
    )
    return parser.parse_args()

def wilcoxon_test(x, y, show=True):
    result = wilcoxon(x, y)
    if show:
        print(result)
    return result

def student_test(x, y):
    print(ttest_ind(x, y))

#calculate mean and std for metrics
def mean_and_std(all_values, output, run_test=False):
    mean = {}
    std = {}
    for i in list(all_values.keys()):
        mean[i] = np.mean(all_values[i])
        std[i] = np.std(all_values[i])
        line = f'{str(i)},{str(mean[i]*100)},{str(std[i])}'
        if run_test:
            if not i == "no_weight":
                x = all_values["no_weight"]
                y = all_values[i]
                test_result = wilcoxon_test(x, y, False)
                line = f'{line},p={test_result[1]}'
        print(line)
        output.write(f'{line}\n')

    return mean, std

#plot mean and std for each strategy and metric
def plot_graphic(strategies, mean_acc, std_acc, mean_f1, std_f1, dataset, window, cut_percent, output_fig):
    plt.style.use('seaborn-notebook')
    plt.style.use('seaborn')
    legend_map = {
        "no_weight": "SEM PESO",
        "pmi": "PMI LOCAL",
        "pmi_all": "PMI GLOBAL",
        "llr": "LLR LOCAL",
        "llr_all": "LLR GLOBAL",
        "chi_square": "CS LOCAL",
        "chi_square_all": "CS GLOBAL"
    }

    bar = []
    acc_mean = []
    acc_std = []
    f1_mean = []
    f1_std = []
    for s in strategies:
        bar.append(legend_map[s])
        acc_mean.append(mean_acc[s])
        acc_std.append(std_acc[s])
        f1_mean.append(mean_f1[s])
        f1_std.append(std_f1[s])

    positions = np.arange(len(bar))
    width = 0.3
    fig, ax = plt.subplots()
    ax.bar(positions - (width/2), acc_mean, width, yerr=acc_std, capsize=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(bar)

    ax.bar(positions + (width/2), f1_mean, width, yerr=f1_std, capsize=5)
    plt.setp(ax.get_xticklabels(), rotation='35', horizontalalignment='right')
    plt.xlabel("Medida de associatividade")
    plt.ylabel("Valor da taxa de acerto e do F1-score")
    plt.title("Dataset: {0} Janela: {1} Corte: {2}%".format(dataset, window, cut_percent))
    plt.legend(["Taxa de acerto", "F1-score"], loc='upper left')
    plt.tight_layout()
    plt.savefig(output_fig)
    plt.close()

#run statistical tests in betwens accuracy and f1 score strategies values
# Test 1: wilcoxon test
# Test 2: student test
def statistical_tests(all_acc, all_f1, window, strategies):
    print("===== STATISTICAL TESTS =====")
    it = strategies[1:]
    print(" -- TESTING ACCURACY -- ")
    x = all_acc["no_weight"]
    for i in it:
        print(i)
        y = all_acc[i]
        wilcoxon_test(x, y)
        student_test(x, y)

    print(" -- TESTING F1-SCORE -- ")
    x = all_f1["no_weight"]
    for i in it:
        print(i)
        y = all_f1[i]
        wilcoxon_test(x, y)
        student_test(x, y)

def make_results_dir(directory):
    if not os.path.exists(directory):
        print("results dir not exist, creating ...")
        os.makedirs(directory)

    return directory

def main():
    args = read_args()

    cut_percent = args.cut_percent/100.0 
    method = "node2vec"

    directory = (
        "results/next_level/" +
        args.dataset +
        "-" +
        str(args.emb_dim) +
        "/0.0/"
    )

    all_acc = {}
    all_f1 = {}

    s = "no_weight"
    file_all_acc_no_weight = (
        directory + args.dataset + '_' + method + '_' + s + '_' + str(args.window) + '.txt'
    )
    f = open(file_all_acc_no_weight, 'r')
    all_acc[s] = np.array([line.rstrip('\n') for line in f]).astype(np.float)
    file_all_ff_no_weight = (
        directory + 'f1_' + args.dataset + '_' + method + '_' + s + '_' + str(args.window) + '.txt'
    )
    ff = open(file_all_ff_no_weight, 'r')
    all_f1[s] = np.array([line.rstrip('\n') for line in ff]).astype(np.float)

    directory = (
        "results/next_level/" +
        args.dataset +
        "-" +
        str(args.emb_dim) +
        "/" +
        str(cut_percent) +
        "/"
    )

    default_output_dir = make_results_dir("plots/next_level/" + str(cut_percent) + "/" + args.dataset) 
    default_output = default_output_dir + "/" + args.dataset + "_" + str(args.window) + ".txt"
    acc_output = open(default_output, "w")
    f1_output = open(
        file=default_output_dir  + "/" + "f1_" + args.dataset + "_" + str(args.window) + ".txt",
        mode='w'
    )
    output_fig = default_output + ".png"

    strategies = ["chi_square", "chi_square_all", "llr", "llr_all", "pmi", "pmi_all"]
    #read results for each strategy
    for s in strategies:
        f = open(
            file=directory + args.dataset + '_' + method + '_' + s + '_' + str(args.window) + '.txt',
            mode='r'
        )
        all_acc[s] = np.array([line.rstrip('\n') for line in f]).astype(np.float)
        file_all_acc_strategy = (
            directory + 'f1_' + args.dataset + '_' + method
            + '_' + s + '_' + str(args.window) + '.txt'
        )
        ff = open(
            file=file_all_acc_strategy,
            mode='r'
        )
        all_f1[s] = np.array([line.rstrip('\n') for line in ff]).astype(np.float)

    no_wight_directory = (
        "results/" +
        args.dataset +
        "-" +
        str(args.emb_dim) +
        "/"
    )

    strategies = ["no_weight", "pmi", "pmi_all", "llr", "llr_all", "chi_square", "chi_square_all"]
    print('==== ACCURACY ====')
    mean_acc, std_acc = mean_and_std(all_acc, acc_output)
    print('==== F1 SCORE ====')
    mean_f1, std_f1 = mean_and_std(all_f1, f1_output, True)
    statistical_tests(all_acc, all_f1, args.window, strategies)

    plot_graphic(
        strategies=strategies,
        mean_acc=mean_acc,
        std_acc=std_acc,
        mean_f1=mean_f1,
        std_f1=std_f1,
        dataset=args.dataset,
        window=args.window,
        cut_percent=int(cut_percent*100),
        output_fig=output_fig
    )

if __name__ == "__main__":
    main()
