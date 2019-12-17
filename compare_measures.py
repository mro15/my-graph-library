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
    return parser.parse_args()

def get_legend(strategy):
    legend_map = { "no_weight": "Sem peso",
                "pmi_1990": "PMI (1990)",
                "pmi_1990_all": "PMI (1990) ALL"}
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

def plot_edges(x, method_x, y, method_y):
    aux = np.arange(1, len(x)+1)
    fig = plt.figure(figsize=(80, 20))
    ax = fig.add_subplot(111)
    ax.plot(aux, x, 'ro-', color='green')
    ax.plot(aux, y, 'ro-', color='blue')
    plt.savefig("line_"+method_x+"_"+method_y+".png")

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
    plot_cost(args.strategy, edges, args.window, args.dataset, bar)   
    #plot boxplot with the amount of edges
    plot_boxplot([edges[s] for s in args.strategy], bar, args.dataset+"_number_of_edges_"+str(args.window))
    #get number of edges that are small than the original graph
    for s in args.strategy[1::]:
        edges_sub(edges["no_weight"], edges[s], s)
    #density of each graph
    density = {}
    for s in args.strategy:
        density[s] = measure_density(graphs[s]["train"], graphs[s]["test"])
    #plot boxplot with the graph density
    plot_boxplot([density[s] for s in args.strategy], bar, args.dataset+"_density_"+str(args.window))

if __name__ == "__main__":
    main()

