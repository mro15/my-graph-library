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
    parser.add_argument('--dataset', type=str, choices=["imdb", "polarity", "mr", "webkb"], help='dataset name', required=True)   
    parser.add_argument('--window', type=int,  help='window size', required=True)
    parser.add_argument('--strategy', action="append", help='methods to compare', required=True)
    return parser.parse_args()

def get_legend(strategy):
    legend_map = { "no_weight": "Sem peso",
                "pmi_1990": "PMI (1990)",
                "pmi_1990_all": "PMI (1990) ALL"}
    return legend_map[strategy]

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

def plot_cost(strategies, edges, window):
    mean, std = mean_and_std(edges, strategies)
    print(mean, std)
    
    mean_v = []
    std_v = []
    bar = []
    for s in strategies:
        mean_v.append(mean[s])
        std_v.append(std[s])
        bar.append(get_legend(s))
    plt.errorbar(bar, mean_v, yerr=std_v, marker='s', capsize=5)
    plt.xlabel("Métricas")
    plt.ylabel("# arestas")
    plt.title("Comparação # arestas médio," + " #janela: " + str(window))
    plt.savefig("analysis/cost_mean_edges_" + str(window) + ".png")
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
    else:
        print("Error: dataset name unknown")
        return 1

    print("PRE PROCESS: START")
    d.pre_process_data()
    print("PRE PROCESS: END")

    strategies_map = { "no_weight": utils.graph_strategy_one,
                        "pmi_1990": utils.graph_strategy_three,
                        "pmi_1990_all": utils.graph_strategy_three_all,
                        "pmi_2019": utils.graph_strategy_two,
                        "llr": utils.graph_strategy_five,
                        "llr_all": utils.graph_strategy_five_all }
    graphs = {}
    for s in args.strategy:
        graphs[s] = {}
        edges_train, edges_test = strategies_map[s](d, args.window)
        graphs[s]["train"] = edges_train
        graphs[s]["test"] = edges_test

    #dice_edges_train, dice_edges_test = utils.graph_strategy_four(d, args.window)
    #dice_all_edges_train, dice_all_edges_test = utils.graph_strategy_four_all(d, args.window)
    #llr_all_edges_train, llr_all_edges_test = utils.graph_strategy_five_all(d, args.window)
    #chi_square_edges_train, chi_square_edges_test = utils.graph_strategy_six(d, args.window)
    #chi_square_all_edges_train, chi_square_all_edges_test = utils.graph_strategy_six_all(d, args.window)

    #number of edges of each graph
    edges = {}
    for s in args.strategy:
        edges[s] = count_edges(graphs[s]["train"], graphs[s]["test"])

    #edges_dice = count_edges(dice_edges_train, dice_edges_test)
    #edges_dice_all = count_edges(dice_all_edges_train, dice_all_edges_test)
    #edges_llr = count_edges(llr_edges_train, llr_edges_test)
    #edges_llr_all = count_edges(llr_all_edges_train, llr_all_edges_test)
    #edges_chi_square = count_edges(chi_square_edges_train, chi_square_edges_test)
    #edges_chi_square_all = count_edges(chi_square_all_edges_train, chi_square_all_edges_test)
    #plot_boxplot([edges_all, edges_pmi_2019, edges_pmi_1990, edges_pmi_1990_all, edges_dice, edges_dice_all, edges_llr, edges_llr_all, edges_chi_square, edges_chi_square_all], ["Sem peso", "PMI (2019)", "PMI (1990)", "PMI (1990) all", "Dice", "Dice all", "LLR", "LLR all", "Chi-square", "Chi-square all"], args.dataset+"_number_of_edges_"+str(args.window))


    plot_cost(args.strategy, edges, args.window)

    exit()
    #get number of edges that are small
    edges_sub(edges_all, edges_pmi_2019, "pmi_2019")
    edges_sub(edges_all, edges_pmi_1990, "pmi_1990")
    edges_sub(edges_all, edges_pmi_1990_all, "pmi_1990_all")
    edges_sub(edges_all, edges_dice, "dice")
    edges_sub(edges_all, edges_dice_all, "dice_all")
    edges_sub(edges_all, edges_llr, "llr")
    edges_sub(edges_all, edges_llr_all, "llr_all")
    edges_sub(edges_all, edges_chi_square, "chi_square")
    edges_sub(edges_all, edges_chi_square_all, "chi_square_all")

    #density of each graph
    density_all = measure_density(all_edges_train, all_edges_test)
    density_pmi_2019 = measure_density(pmi_2019_edges_train, pmi_2019_edges_test)
    density_pmi_1990 = measure_density(pmi_1990_edges_train, pmi_1990_edges_test)
    density_pmi_1990_all = measure_density(pmi_1990_all_edges_train, pmi_1990_all_edges_test)
    density_dice = measure_density(dice_edges_train, dice_edges_test)
    density_dice_all = measure_density(dice_all_edges_train, dice_all_edges_test)
    density_llr = measure_density(llr_edges_train, llr_edges_test)
    density_llr_all = measure_density(llr_all_edges_train, llr_all_edges_test)
    density_chi_square = measure_density(chi_square_edges_train, chi_square_edges_test)
    density_chi_square_all = measure_density(chi_square_all_edges_train, chi_square_all_edges_test)
    plot_boxplot([density_all, density_pmi_2019, density_pmi_1990, density_pmi_1990_all, density_dice, density_dice_all, density_llr, density_llr_all, density_chi_square,  density_chi_square_all], ["Sem peso", "PMI (2019)", "PMI (1990)",  "PMI (1990) all",  "Dice", "Dice all", "LLR", "LLR all", "Chi-square", "Chi-square all"], args.dataset+"_density_"+str(args.window))

if __name__ == "__main__":
    main()

