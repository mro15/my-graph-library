#! /usr/bin/env python3

import argparse
import numpy as np
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--dataset', type=str, choices=["imdb", "polarity", "mr"], help='dataset name', required=True)   
    parser.add_argument('--method', type=str, choices=["node2vec", "gcn"], help='representation method', required=True)
    return parser.parse_args()

def mean_and_std(res, strat, windows, dataset, output):
    means = {}
    stds = {}
    for s in strat:
        for w in windows:
            for v in res[s]:
                m = np.mean(v[w])
                sd = np.std(v[w])
                line = s, str(w)+" =>", "mean: "+str(m), "std: "+str(sd)
                print(line)
                output.write(str(line)+"\n")
                means[s]=m
                stds[s]=sd
    plot_graphic(windows, strat, means, stds, dataset)

def plot_graphic(windows, strat, means, stds, dataset):
    for s in strat:
        acc = []
        win = []
        sd = []
        for w in windows:
            acc.append(means[s])
            win.append(w)
            sd.append(stds[s])
        print(win, acc, sd)
        plt.errorbar(win, acc, yerr=sd)
    plt.legend(["no_weight", "pmi"], loc="upper left")
    plt.xlabel("window size")
    plt.ylabel("accuracy")
    plt.savefig("plots/"+dataset+".png")
    plt.close()

def main():
    args = read_args()

    strategies = ["no_weight", "pmi"]
    windows = [4]
    all_res = {"no_weight":[], "pmi":[]}
    output = open("plots/"+args.dataset+".txt", "w")
    for s in strategies:
        for w in windows:
            f = open('results/' + args.dataset + '_' + args.method + '_' + s + '_' + str(w) + '.txt', 'r')
            all_res[s].append({w:np.array([line.rstrip('\n') for line in f]).astype(np.float)})
    for s in strategies:
        for w in windows:
            for v in all_res[s]:
                print(s, w, v[w])
    mean_and_std(all_res, strategies, windows, args.dataset, output)

if __name__ == "__main__":
    main()
