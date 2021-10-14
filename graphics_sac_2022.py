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


from text_handler.dataset import Dataset
from weight_cutter.weight_cutter import WeightCutter

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
    return parser.parse_args()


#plot mean and std for each strategy and metric
def plot_f1_score(cuts, strategies, mean_f1, std_f1, dataset, window,output_fig):
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    legend_map = {
        "no_weight": "UNWEIGHTED",
        "pmi": "LOCAL PMI",
        "pmi_all": "GLOBAL PMI",
        "llr": "LOCAL LLR",
        "llr_all": "GLOBAL LLR",
        "chi_square": "LOCAL CHI-SQUARE",
        "chi_square_all": "GLOBAL CHI-SQUARE"
    }

    for s in mean_f1:
        line_f1 = []
        std_f1_bar = []
        line_cut = []
        for c in mean_f1[s]:
            line_f1.append(mean_f1[s][str(c)])
            std_f1_bar.append(std_f1[s][str(c)])
            line_cut.append(c)
        plt.errorbar(
            x=line_cut, y=line_f1, yerr=std_f1_bar, fmt='.',
            label=legend_map[s], capsize=8, linewidth=1, linestyle='--',
            capthick=1
        )

    ax.legend()
    plt.xlabel("CUT PERCENTAGE")
    plt.ylabel("F1-SCORE")
    plt.tight_layout()
    plt.savefig(output_fig)
    plt.close()


#calculate mean and std for metrics
def mean_and_std(all_values):
    mean = {}
    std = {}
    for s in all_values:
        mean[s] = {}
        std[s] = {}
        for c in all_values[s]:
            mean[s][c] = np.mean(all_values[s][c])
            std[s][c] = np.std(all_values[s][c])

    return mean, std

def build_graphs(dataset, window, strategy, cut_percentage, emb_dim):
    weight_cutter = WeightCutter(
        emb_dim=emb_dim,
        dataset=dataset,
        strategy=strategy,
        window_size=window,
        cut_percentage=cut_percentage
    )

    weight_cutter.construct_graphs()
    return weight_cutter

def read_dataset(dataset):
    d = Dataset(dataset)
    dataset_readers={
        "polarity": "read_polarity",
        "webkb": "read_webkb",
        "r8": "read_r8",
        "20ng": "read_20ng"
    }
    read_function = getattr(d, dataset_readers.get(dataset))
    read_function()
    return d


def vertex_and_edges(dataset, window, emb_dim, cut, strategy):

    weight_cutter = build_graphs(
        dataset=dataset,
        window=window,
        strategy=strategy,
        cut_percentage=cut*100,
        emb_dim=emb_dim
    )

    graphs = (
            weight_cutter.graph_builder.train_graphs +
            weight_cutter.graph_builder.test_graphs
    )
    print(len(graphs))
    edges = [amount.number_of_edges() for amount in graphs]
    nodes = [amount.number_of_nodes() for amount in graphs]
    print(f'==== CUT: {cut} =====')
    print('EDGES')
    print(f'LEN: {len(edges)}, MEAN: {np.mean(edges)}, STD: {np.std(edges)}')
    print('NODES')
    print(f'LEN: {len(nodes)}, MEAN: {np.mean(nodes)}, STD: {np.std(nodes)}')
    weight_cutter = None
    graphs = None
    return np.mean(edges), np.mean(nodes)


def main():
    args = read_args()

    method = "node2vec"

    dataset = args.dataset 

    output_fig = "sac_results/" + dataset + "_" + str(args.window) + ".png"
    strategies = ["no_weight", "chi_square", "chi_square_all", "llr", "llr_all", "pmi", "pmi_all"]
    all_f1 = {}
    for s in strategies:
        all_f1[s] = {}

    directory = (
        "results/next_level/" +
        dataset +
        "-" +
        str(args.emb_dim) +
        "/0.0/"
    )

    file_all_ff_no_weight = (
        directory + 'f1_' + dataset + '_' + method + '_no_weight_' + str(args.window) + '.txt'
    )
    ff = open(file_all_ff_no_weight, 'r')
    all_f1['no_weight']['0'] = np.array([line.rstrip('\n') for line in ff]).astype(np.float)

    cuts = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

    #read results for each strategy
    for s in strategies[1::]:
        for c in cuts[1::]:
            directory = (
                "results/next_level/" +
                dataset +
                "-" +
                str(args.emb_dim) +
                "/" +
                str(c) +
                "/"
            )

            file_all_f1_strategy = (
                directory + 'f1_' + dataset + '_' + method
                + '_' + s + '_' + str(args.window) + '.txt'
            )
            ff = open(
                file=file_all_f1_strategy,
                mode='r'
            )
            all_f1[s][str(c)] = np.array([line.rstrip('\n') for line in ff]).astype(np.float)

    mean_f1, std_f1 = mean_and_std(all_f1)


    plot_f1_score(
        cuts=cuts,
        strategies=strategies,
        mean_f1=mean_f1,
        std_f1=std_f1,
        dataset=dataset,
        window=args.window,
        output_fig=output_fig
    )

if __name__ == "__main__":
    main()

