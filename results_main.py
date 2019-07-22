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
    parser.add_argument('--dataset', type=str, choices=["imdb", "polarity", "mr"], help='dataset name', required=True)   
    parser.add_argument('--method', type=str, choices=["node2vec", "gcn"], help='representation method', required=True)
    parser.add_argument('--emb_dim)', type=int, help='embeddings dimension', required=True)
    return parser.parse_args()

def wilcoxon_test(x, y):
    print(wilcoxon(x, y))

def student_test(x, y):
    print(ttest_ind(x, y))

def mean_and_std(res, strat, windows, dataset, output):
    means = {}
    stds = {}
    for s in strat:
        m_l = []
        s_l = []
        for w in range(0, len(windows)):
            m = np.mean(res[s][w])
            sd = np.std(res[s][w])
            line = s, str(windows[w])+" =>", "mean: "+str(m), "std: "+str(sd)
            print(line)
            output.write(str(line)+"\n")
            m_l.append(m)
            s_l.append(sd)
        means[s]=m_l
        stds[s]=s_l
    plot_graphic(windows, strat, means, stds, dataset, output)

def plot_graphic(windows, strat, means, stds, dataset, output):
    for s in strat:
        print(windows, means[s], stds[s])
        plt.errorbar(windows, means[s], yerr=stds[s], fmt='o', marker='s', capsize=10)
    plt.legend(["no_weight", "pmi", "normalized_pmi"], loc="upper left", numpoints=1)
    plt.xlabel("window size")
    plt.ylabel("accuracy")
    plt.xlim(windows[0]-2, windows[-1]+2)
    plt.savefig(output + ".png")
    plt.close()

def main():
    args = read_args()

    directory = args.dataset + "-" + str(args.emb_dim) + "/"
    strategies = ["no_weight", "pmi", "normalized_pmi"]
    windows = [4, 5, 7]
    all_res = {"no_weight":[], "pmi":[], "normalized_pmi":[]}
    output = open("plots/" + directory + args.dataset+".txt", "w")
    for s in strategies:
        for w in windows:
            f = open("results/" + directory + args.dataset + '_' + args.method + '_' + s + '_' + str(w) + '.txt', 'r')
            all_res[s].append(np.array([line.rstrip('\n') for line in f]).astype(np.float))
    for s in strategies:
        for w in range(0, len(windows)):
            print(all_res[s][w])
        print("---")

    for w in range(0, len(windows)):
        y = all_res["no_weight"][w]
        x = all_res["pmi"][w]
        print("=== NO_WEIGHT & PMI ===")
        wilcoxon_test(x, y)
        student_test(x, y)
        x = all_res["normalized_pmi"][w]
        print("=== NO_WEIGHT & NORMALIZED_PMI ===")
        wilcoxon_test(x, y)
        student_test(x, y)
    mean_and_std(all_res, strategies, windows, args.dataset, output)

if __name__ == "__main__":
    main()
