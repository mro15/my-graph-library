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
    parser.add_argument('--emb_dim', type=int, help='embeddings dimension', required=True)
    return parser.parse_args()

def wilcoxon_test(x, y):
    print(wilcoxon(x, y))

def student_test(x, y):
    print(ttest_ind(x, y))

def mean_and_std(res, strat, windows, dataset, output, output_fig):
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
    plot_graphic(windows, strat, means, stds, dataset, output_fig)

def plot_graphic(windows, strat, means, stds, dataset, output_fig):
    for s in strat:
        print(windows, means[s], stds[s])
        plt.errorbar(windows, means[s], yerr=stds[s], fmt='o', marker='s', capsize=10)
    plt.legend(strat, loc="upper left", numpoints=1)
    plt.xlabel("Tamanho da janela")
    plt.ylabel("Taxa de acerto")
    plt.xlim(windows[0]-2, windows[-1]+2)
    plt.savefig(output_fig + ".png")
    plt.close()

def mean_and_std_w(res, strat, windows, dataset, output, output_fig):
    means = {}
    stds = {}
    for w in range(0, len(windows)):
        m_l = []
        s_l = []
        for s in strat:
            m = np.mean(res[s][w])
            sd = np.std(res[s][w])
            line = s, str(windows[w])+" =>", "mean: "+str(m), "std: "+str(sd)
            print(line)
            output.write(str(line)+"\n")
            m_l.append(m)
            s_l.append(sd)
        means[w]=m_l
        stds[w]=s_l
    plot_graphic_w(windows, strat, means, stds, dataset, output_fig)

def plot_graphic_w(windows, strat, means, stds, dataset, output_fig):
    bar =  ["Sem peso", "PMI (1990)", "PMI (2019)", "Dice", "LLR", "Chi-square"]
    for w in range(0, len(windows)):
        title = "Tamanho da janela: " + str(windows[w])
        print("---")
        print(windows[w], means[w], stds[w])
        plt.errorbar(bar, means[w], yerr=stds[w], marker='s', capsize=10)
        plt.xlabel("Métrica")
        plt.ylabel("Taxa de acerto")
        plt.title(title)
        plt.savefig(output_fig + str(windows[w]) + ".png")
        plt.close()


def main():
    args = read_args()
    print(args.emb_dim)

    directory = args.dataset + "-" + str(args.emb_dim) + "/"
    strategies = ["no_weight", "pmi_1990", "pmi_2019", "dice", "llr", "chi_square"]
    windows = [4, 7, 12, 20]
    all_res = {"no_weight":[], "pmi_1990":[], "pmi_2019":[], "dice":[], "llr":[], "chi_square":[]}
    output = open("plots/" + directory + args.dataset+".txt", "w")
    output_fig = "plots/" + directory + args.dataset+".txt"
    print("===== ACCURACY =====")
    for s in strategies:
        for w in windows:
            f = open("results/" + directory + args.dataset + '_' + args.method + '_' + s + '_' + str(w) + '.txt', 'r')
            all_res[s].append(np.array([line.rstrip('\n') for line in f]).astype(np.float))
    for s in strategies:
        for w in range(0, len(windows)):
            print(all_res[s][w])
        print("---")

    for w in range(0, len(windows)):
        print("WINDOW: ", windows[w])
        y = all_res["no_weight"][w]
        x = all_res["pmi_2019"][w]
        print("=== NO_WEIGHT & PMI (2019) ===")
        wilcoxon_test(x, y)
        student_test(x, y)
        """
        x = all_res["normalized_pmi"][w]
        print("=== NO_WEIGHT & NORMALIZED_PMI ===")
        wilcoxon_test(x, y)
        student_test(x, y)
        """
        x = all_res["pmi_1990"][w]
        print("=== NO_WEIGHT & PMI (1990) ===")
        wilcoxon_test(x, y)
        student_test(x, y)
        x = all_res["dice"][w]
        print("=== NO_WEIGHT & DICE ===")
        wilcoxon_test(x, y)
        student_test(x, y)
        x = all_res["llr"][w]
        print("=== NO_WEIGHT & LRR ===")
        wilcoxon_test(x, y)
        student_test(x, y)
        x = all_res["chi_square"][w]
        print("=== NO_WEIGHT & Chi Square ===")
        wilcoxon_test(x, y)
        student_test(x, y)
    mean_and_std_w(all_res, strategies, windows, args.dataset, output, output_fig)

    print("===== F1-SCORE =====")
    output = open("plots/" + directory + "f1_" + args.dataset+".txt", "w")
    output_fig = "plots/" + directory + "f1_" + args.dataset+".txt"
    for s in strategies:
        for w in windows:
            f = open("results/" + directory + "f1_" + args.dataset + '_' + args.method + '_' + s + '_' + str(w) + '.txt', 'r')
            all_res[s].append(np.array([line.rstrip('\n') for line in f]).astype(np.float))
    for s in strategies:
        for w in range(0, len(windows)):
            print(all_res[s][w])
        print("---")
    mean_and_std_w(all_res, strategies, windows, args.dataset, output, output_fig)


if __name__ == "__main__":
    main()
