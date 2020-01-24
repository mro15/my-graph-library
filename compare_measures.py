#! /usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import utils
from text_graph.text_graph import TextGraph
from text_graph.node_features import NodeFeatures
from text_handler.dataset import Dataset
import numpy as np
import networkx as nx

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--dataset', type=str, choices=["imdb", "polarity", "mr", "webkb", "ohsumed", "20ng"], help='dataset name', required=True)   
    parser.add_argument('--window', type=int,  help='window size', required=True)
    parser.add_argument('--strategy', action="append", help='methods to compare', required=True)
    parser.add_argument('--emb_dim', type=int, help='embeddings dimension', required=True)
    #parser.add_argument('--method', type=str, choices=["node2vec", "gcn"], help='representation method', required=True)
    return parser.parse_args()

def get_legend(strategy):
    legend_map = {  "no_weight": "UNWEIGHTED",
                    "pmi_1990": "LOCAL PMI",
                    "pmi_1990_all": "GLOBAL PMI"}
    return legend_map[strategy]

def strategies_to_bar(strategies):
    bar = [get_legend(s) for s in strategies]
    return bar

def plot_boxplot(values, methods, name):
    fig1, ax1 = plt.subplots()
    ax1.boxplot(values)
    ax1.set_xticklabels(methods)
    ax1.yaxis.grid(True)
    plt.setp(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig("analysis/box_plot_"+name+".png")
    plt.close()

def plot_histogram(x, method_x, y, method_y):
    plt.title(method_x+" x "+method_y)
    plt.xlabel('# edges')
    plt.ylabel('# graphs')
    plt.hist([x, y], histtype='bar')
    plt.legend([method_x, method_y])
    plt.savefig("histogram_"+method_x+"_"+method_y+".png")
    plt.close()

def measure_density(g_train, g_test):
    density = []
    for i in g_train:
        density.append(nx.density(i))
    for i in g_test:
        density.append(nx.density(i))
    return density

def count_edges(g_train, g_test):
    count = []
    for i in g_train:
        count.append(i.number_of_edges())
    for i in g_test:
        count.append(i.number_of_edges())
    return count

def edges_sub(a, b, method):
    sub = np.array(a) - np.array(b)
    cont = 0
    number = 0
    for i in sub:
        if not i == 0:
            cont+=1
            number += i
    print(method, cont, number)
    return sub

def proportion(edges, strategies):
    proportions = {}
    #get number of edges that are small than the original graph
    for s in strategies:
        sub = edges_sub(edges["no_weight"], edges[s], s)
        arr_div = np.array(sub/edges["no_weight"])
        arr_div[np.isnan(arr_div)] = 0
        proportions[s] = np.mean(arr_div*100)
        print(proportions[s])
    return proportions 

#calculate mean and std for metrics
def mean_and_std(edges, strategies):
    mean = {}
    std = {}
    for s in strategies:
        mean[s] = np.mean(edges[s])
        std[s] = np.std(edges[s])
    return mean, std

#plot cost (amount of edges: mean and std)
def plot_cost(strategies, edges, window, dataset, bar):
    mean, std = mean_and_std(edges, strategies)
    print(mean, std)
    
    mean_v = []
    std_v = []
    for s in strategies:
        mean_v.append(mean[s])
        std_v.append(std[s])
    plt.errorbar(bar, mean_v, yerr=std_v, marker='s', capsize=5)
    plt.xlabel("Métricas")
    plt.ylabel("# arestas")
    plt.title(dataset + ": Comparação # arestas médio," + " #janela: " + str(window))
    plt.savefig("analysis/" + dataset + "_cost_mean_edges_" + str(window) + ".png")
    plt.close()


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    return ax

def plot_cost_benefit(proportions, fscores, bar, strategies, output):
    x = np.arange(len(bar))
    width = 0.25
    edges = []
    acc = []
    for s in strategies:
        edges.append(round(float(proportions[s]), 3))
        acc.append(round(float(fscores[s])*100, 3))

    color = '#2e5a88'
    fig, ax = plt.subplots()
    ax.set_ylabel('F1-score', color=color)
    f_bar = ax.bar(x - width/2, acc, width, label='F1-score', color=color)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xticks(x)
    ax.set_xticklabels(bar, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.tick_params(axis='y', labelcolor=color)

    ax2 = ax.twinx()
    color = '#850e04'
    ax2.set_ylabel('% removed edges', color=color)
    edges_bar = ax2.bar(x + width/2, edges, width, label='% removed edges', color=color)
    ax2.set_yticks(np.arange(0, 101, 10))
    ax2.tick_params(axis='y', labelcolor=color)

    ax = autolabel(f_bar, ax)
    ax2 = autolabel(edges_bar, ax2)

    fig.tight_layout()
    plt.savefig("analysis/"+output+".eps")
    plt.close()

def main():
    args = read_args()
    if args.dataset == "polarity":
        d = Dataset(args.dataset)
        d.read_polarity()
    elif args.dataset == "imdb":
        d = Dataset(args.dataset)
        d.read_imdb()
    elif args.dataset == "mr":
        d = Dataset(args.dataset)
        d.read_mr()
    elif args.dataset == "webkb":
        d = Dataset(args.dataset)
        d.read_webkb()
    elif args.dataset == "ohsumed":
        d = Dataset(args.dataset)
        d.read_ohsumed()
    elif args.dataset == "20ng":
        d = Dataset(args.dataset)
        d.read_20ng()
    else:
        print("Error: dataset name unknown")
        return 1
    d.pre_process_data()

    strategies_map = { "no_weight": utils.graph_strategy_one,
                        "pmi_1990": utils.graph_strategy_three,
                        "pmi_1990_all": utils.graph_strategy_three_all,
                        "pmi_2019": utils.graph_strategy_two,
                        "llr": utils.graph_strategy_five,
                        "llr_all": utils.graph_strategy_five_all,
                        "chi_square": utils.graph_strategy_six,
                        "chi_square_all": utils.graph_strategy_six_all }

    bar = strategies_to_bar(args.strategy)
    #build graphs using each strategy
    graphs = {}
    for s in args.strategy:
        graphs[s] = {}
        edges_train, edges_test = strategies_map[s](d, args.window)
        graphs[s]["train"] = edges_train
        graphs[s]["test"] = edges_test
    #number of edges of each graph
    edges = {}
    for s in args.strategy:
        edges[s] = count_edges(graphs[s]["train"], graphs[s]["test"])
    #plot_cost(args.strategy, edges, args.window, args.dataset, bar)   
    #plot boxplot with the amount of edges
    plot_boxplot([edges[s] for s in args.strategy], bar, args.dataset+"_number_of_edges_"+str(args.window))
    #density of each graph
    density = {}
    for s in args.strategy:
        density[s] = measure_density(graphs[s]["train"], graphs[s]["test"])
    #plot boxplot with the graph density
    plot_boxplot([density[s] for s in args.strategy], bar, args.dataset+"_density_"+str(args.window))
    
    #graph for cost x benefit
    proportions = proportion(edges, args.strategy)
    mean_f1 = {}
    f = open("plots/" + args.dataset + '-' + str(args.emb_dim) + '/f1_' + args.dataset + "_" + str(args.window) + '.txt', 'r')
    lines = f.readlines()
    for line in lines:
        x = line.strip().split(",")
        mean_f1[x[0]] = x[1]
    print(mean_f1)
    plot_cost_benefit(proportions, mean_f1, bar, args.strategy, args.dataset+"_cost_"+str(args.window))

if __name__ == "__main__":
    main()

