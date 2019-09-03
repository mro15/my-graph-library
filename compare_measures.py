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
    parser.add_argument('--dataset', type=str, choices=["imdb", "polarity", "mr"], help='dataset name', required=True)   
    parser.add_argument('--window', type=int,  help='window size', required=True)
    return parser.parse_args()

def plot_boxplot(values, methods, name):
    fig1, ax1 = plt.subplots()
    ax1.boxplot(values)
    ax1.set_xticklabels(methods)
    ax1.yaxis.grid(True)
    plt.savefig("box_plot_"+name+".png")
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
    else:
        print("Error: dataset name unknown")
        return 1
    print("PRE PROCESS: START")
    d.pre_process_data()
    print("PRE PROCESS: END")

    all_edges_train, all_edges_test = utils.graph_strategy_two(d, args.window)
    pmi_2019_edges_train, pmi_2019_edges_test = utils.graph_strategy_three(d, args.window)
    pmi_1990_edges_train, pmi_1990_edges_test = utils.graph_strategy_five(d, args.window)
    llr_edges_train, llr_edges_test = utils.graph_strategy_seven(d, args.window)
    dice_edges_train, dice_edges_test = utils.graph_strategy_six(d, args.window)
    chi_square_edges_train, chi_square_edges_test = utils.graph_strategy_six(d, args.window)

    #number of edges of each graph
    edges_all = count_edges(all_edges_train, all_edges_test)
    edges_pmi_2019 = count_edges(pmi_2019_edges_train, pmi_2019_edges_test)
    edges_pmi_1990 = count_edges(pmi_1990_edges_train, pmi_1990_edges_test)
    edges_llr = count_edges(llr_edges_train, llr_edges_test)
    edges_dice = count_edges(dice_edges_train, dice_edges_test)
    edges_chi_square = count_edges(chi_square_edges_train, chi_square_edges_test)
    plot_boxplot([edges_all, edges_pmi_2019, edges_pmi_1990, edges_llr, edges_dice, edges_chi_square], ["no_weight", "pmi_2019", "pmi_1990", "llr", "dice", "chi_square"], "number_of_edges_"+str(args.window))

    #get number of edges that are small
    edges_sub(edges_all, edges_pmi_2019, "pmi_2019")
    edges_sub(edges_all, edges_pmi_1990, "pmi_1990")
    edges_sub(edges_all, edges_llr, "llr")
    edges_sub(edges_all, edges_dice, "dice")
    edges_sub(edges_all, edges_chi_square, "chi_square")


    #density of each graph
    density_all = measure_density(all_edges_train, all_edges_test)
    density_pmi_2019 = measure_density(pmi_2019_edges_train, pmi_2019_edges_test)
    density_pmi_1990 = measure_density(pmi_1990_edges_train, pmi_1990_edges_test)
    density_llr = measure_density(llr_edges_train, llr_edges_test)
    density_dice = measure_density(dice_edges_train, dice_edges_test)
    density_chi_square = measure_density(chi_square_edges_train, chi_square_edges_test)
    plot_boxplot([density_all, density_pmi_2019, density_pmi_1990, density_llr, density_dice, density_chi_square], ["no_weight", "pmi_2019", "pmi_1990", "llr", "dice", "chi_square"], "density_"+str(args.window))

if __name__ == "__main__":
    main()

