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

def wilcoxon_test(x, y):
    print(wilcoxon(x, y))

def student_test(x, y):
    print(ttest_ind(x, y))

#calculate mean and std for metrics
def mean_and_std(all_values, output):
    mean = {}
    std = {}
    for i in list(all_values.keys()):
        mean[i] = np.mean(all_values[i])
        std[i] = np.std(all_values[i])
        line = str(i)+","+str(mean[i]*100)+","+str(std[i])+"\n"
        print(line)
        output.write(line)

    return mean, std

#plot mean and std for each strategy and metric
def plot_graphic(strategies, mean_acc, std_acc, mean_f1, std_f1, dataset, window, output_fig):
    legend_map = {
        "no_weight": "SEM PESO",
        "pmi": "LOCAL PMI",
        "pmi_all": "GLOBAL PMI",
        "freq": "LOCAL FREQUENCY",
        "freq_all": "GLOBAL FREQUENCY"
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

    fig, ax = plt.subplots()
    ax.errorbar(bar, acc_mean, yerr=acc_std, marker='s', capsize=5)
    ax.errorbar(bar, f1_mean, yerr=f1_std, marker='s', capsize=5)
    plt.setp(ax.get_xticklabels(), rotation='35', horizontalalignment='right')
    plt.xlabel("Métricas")
    plt.ylabel("Valor")
    plt.title("Dataset: " + dataset + " #janela: " + str(window))
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
        "results/" +
        args.dataset +
        "-" +
        str(args.emb_dim) +
        "/"
    )

    all_acc = {}
    all_f1 = {}

    s = "no_weight"
    f = open(directory + args.dataset + '_' + method + '_' + s + '_' + str(args.window) + '.txt', 'r')
    all_acc[s] = np.array([line.rstrip('\n') for line in f]).astype(np.float)
    ff = open(directory + 'f1_' + args.dataset + '_' + method + '_' + s + '_' + str(args.window) + '.txt', 'r')
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

    #strategies = ["no_weight", "pmi_1990", "pmi_1990_all", "freq", "freq_all"]
    default_output_dir = make_results_dir("plots/next_level/" + str(cut_percent) + "/" + args.dataset) 
    default_output = default_output_dir + "/" + args.dataset + "_" + str(args.window) + ".txt"
    acc_output = open(default_output, "w")
    f1_output = open(default_output_dir  + "/" + "f1_" + args.dataset + "_" + str(args.window) + ".txt", "w")
    output_fig = default_output + ".png"

    strategies = ["pmi"]
    #read results for each strategy
    for s in strategies:
        f = open(directory + args.dataset + '_' + method + '_' + s + '_' + str(args.window) + '.txt', 'r')
        all_acc[s] = np.array([line.rstrip('\n') for line in f]).astype(np.float)
        ff = open(directory + 'f1_' + args.dataset + '_' + method + '_' + s + '_' + str(args.window) + '.txt', 'r')
        all_f1[s] = np.array([line.rstrip('\n') for line in ff]).astype(np.float)

    no_wight_directory = (
        "results/" +
        args.dataset +
        "-" +
        str(args.emb_dim) +
        "/"
    )

    strategies = ["no_weight", "pmi"]
    statistical_tests(all_acc, all_f1, args.window, strategies)
    mean_acc, std_acc = mean_and_std(all_acc, acc_output)
    mean_f1, std_f1 = mean_and_std(all_f1, f1_output)

    plot_graphic(strategies, mean_acc, std_acc, mean_f1, std_f1, args.dataset, args.window, output_fig)

if __name__ == "__main__":
    main()
