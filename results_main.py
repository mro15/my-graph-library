#! /usr/bin/env python3

import argparse
import numpy as np
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats import ttest_ind

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--dataset', type=str, choices=["imdb", "polarity", "mr", "20ng", "webkb", "ohsumed"], help='dataset name', required=True)   
    parser.add_argument('--method', type=str, choices=["node2vec", "gcn"], help='representation method', required=True)
    parser.add_argument('--emb_dim', type=int, help='embeddings dimension', required=True)
    parser.add_argument('--window', type=int, help='window size', required=True)
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
def plot_graphic(mean_acc, std_acc, mean_f1, std_f1, dataset, window, output_fig):
    legend_map = { "no_weight": "Sem peso",
                "pmi_1990": "PMI (1990)",
                "pmi_1990_all": "PMI (1990) ALL"}
    bar = []
    acc_mean = []
    acc_std = []
    f1_mean = []
    f1_std = []
    for s in mean_acc.keys():
        bar.append(legend_map[s])
        acc_mean.append(mean_acc[s])
        acc_std.append(std_acc[s])
        f1_mean.append(mean_f1[s])
        f1_std.append(std_f1[s])

    plt.errorbar(bar, acc_mean, yerr=acc_std, marker='s', capsize=5)
    plt.errorbar(bar, f1_mean, yerr=f1_std, marker='s', capsize=5)
    plt.xlabel("Métricas")
    plt.ylabel("Valor")
    plt.title("Dataset: " + dataset + " #janela: " + str(window))
    plt.legend(["Taxa de acerto", "F1-score"], loc='upper left')
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
        y = all_acc[i]
        wilcoxon_test(x, y)
        student_test(x, y)

    print(" -- TESTING F1-SCORE -- ")
    x = all_f1["no_weight"]
    for i in it:
        y = all_f1[i]
        wilcoxon_test(x, y)
        student_test(x, y)

def main():
    args = read_args()

    directory = args.dataset + "-" + str(args.emb_dim) + "/"
    strategies = ["no_weight", "pmi_1990", "pmi_1990_all"]
    default_output = "plots/" + directory + args.dataset + "_" + str(args.window) + ".txt"
    acc_output = open(default_output, "w")
    f1_output = open("plots/" + directory + "f1_" + args.dataset + "_" + str(args.window) + ".txt", "w")
    output_fig = default_output + ".png" 
    all_acc = {}
    all_f1 = {}
    #read results for each strategy
    for s in strategies:
        f = open("results/" + directory + args.dataset + '_' + args.method + '_' + s + '_' + str(args.window) + '.txt', 'r')
        all_acc[s] = np.array([line.rstrip('\n') for line in f]).astype(np.float)
        ff = open("results/" + directory + 'f1_' + args.dataset + '_' + args.method + '_' + s + '_' + str(args.window) + '.txt', 'r')
        all_f1[s] = np.array([line.rstrip('\n') for line in ff]).astype(np.float)

    statistical_tests(all_acc, all_f1, args.window, strategies)
    mean_acc, std_acc = mean_and_std(all_acc, acc_output)
    mean_f1, std_f1 = mean_and_std(all_f1, f1_output)
    
    plot_graphic(mean_acc, std_acc, mean_f1, std_f1, args.dataset, args.window, output_fig)

if __name__ == "__main__":
    main()
