#! /usr/bin/env python3

import argparse
from text_graph.text_graph import TextGraph
from text_graph.node_features import NodeFeatures
from text_handler.dataset import Dataset
import utils
import analysis.vocabulary as an
from representation_learning.representation_learning import RepresentationLearning
from vector_representation_learning.rl_bert import MyBert
from features.features import Features
import numpy as np
import _pickle as pickle
import sklearn
from cnn.my_cnn import My_cnn

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--dataset', type=str, choices=["imdb", "polarity", "mr"], help='dataset name', required=True)   
    parser.add_argument('--method', type=str, choices=["node2vec", "gcn"], help='representation method', required=True)
    parser.add_argument('--strategy', type=str, choices=["no_weight", "pmi_2019", "normalized_pmi", "pmi_1990", "dice", "llr", "chi_square"], help='representation method', required=True)
    parser.add_argument('--window', type=int,  help='window size', required=True)
    parser.add_argument('--emb_dim', type=int,  help='embeddings dimension', required=True)
    return parser.parse_args()

#vocabulary analysis
def voc_analysis(d):
    an.all_voc_analysis(d)
    pos_voc = an.pos_voc_analysis(d)
    neg_voc = an.neg_voc_analysis(d)
    an.plot_analysis(d.vocabulary, "words", "count", args.dataset + " all classes", args.dataset + "_all")
    an.plot_analysis(pos_voc, "words", "count", args.dataset + " positive class", args.dataset + "_pos")
    an.plot_analysis(neg_voc, "words", "count", args.dataset + " negative class", args.dataset + "_neg")

#plot some graphs to test
def plot_graphs(train_graphs, test_graphs, size):
    for i in range(0, size):
        utils.plot_graph(train_graphs[i])
    for i in range(0, size):
        utils.plot_graph(test_graphs[i])

def graph_methods(d, method, window_size, strategy, emb_dim):
    print("PRE PROCESS: START")
    d.pre_process_data()
    #for now I will not remove any word
    #d.remove_words()
    print("PRE PROCESS: END")

    #graph construction
    train_graphs = []
    test_graphs = []
    weight = False
    if strategy == "no_weight":
        train_graphs, test_graphs = utils.graph_strategy_two(d, window_size)
    elif strategy == "pmi_2019":
        train_graphs, test_graphs = utils.graph_strategy_three(d, window_size)
        weight = True
    elif strategy == "normalized_pmi":
        train_graphs, test_graphs = utils.graph_strategy_four(d, window_size)
        weight = True
    elif strategy == "pmi_1990":
        train_graphs, test_graphs = utils.graph_strategy_five(d, window_size)
        weight = True
    elif strategy == "dice":
        train_graphs, test_graphs = utils.graph_strategy_six(d, window_size)
        weight = True
    elif strategy == "llr":
        train_graphs, test_graphs = utils.graph_strategy_seven(d, window_size)
        weight = True
    elif strategy == "chi_square":
        train_graphs, test_graphs = utils.graph_strategy_eight(d, window_size)
        weight = True
    else:
        exit()

    train_emb = []
    test_emb = []
    #extract and write graph features
    features_out = Features(d.dataset)
    print("=== STARTING RL IN TRAIN GRAPHS ===")
    for i in range(0, len(train_graphs)):
    #for i in range(0, 10):
        rl = RepresentationLearning(train_graphs[i], method, weight, d.train_data[i], emb_dim)
        rl.initialize_rl_class(window_size)
        rl.representation_method.initialize_model()
        rl.representation_method.train()
        train_emb.append(rl.representation_method.get_embeddings())
    print("=== FINISHED RL IN TRAIN GRAPHS ===")

    print("=== STARTING RL IN TEST GRAPHS ===")
    #for i in range(0, 10):
    for i in range(0, len(test_graphs)):
        rl = RepresentationLearning(test_graphs[i], method, weight, d.test_data[i], emb_dim)
        rl.initialize_rl_class(window_size)
        rl.representation_method.initialize_model()
        rl.representation_method.train()
        test_emb.append(rl.representation_method.get_embeddings())
    print("=== FINISHED RL IN TEST GRAPHS ===")
    return train_emb, test_emb

def vector_methods(d, method):
    bert = MyBert()
    bert.get_embeddings(d.train_data[0])
    bert.get_embeddings(d.train_data[1])


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

    g_methods = ["node2vec"]
    if args.method in g_methods:
        train_emb, test_emb = graph_methods(d, args.method, args.window, args.strategy, args.emb_dim)
    else:
        vector_methods(d, args.method)
   
    print("=== WRITING NODE EMBEDDINGS ===")
    directory = "graphs/" + args.dataset + "-" + str(args.emb_dim) + "/"
    with open(directory + args.dataset + '_' + args.method + '_' + args.strategy + '_' + str(args.window) + '_' + 'train_x.pkl', 'wb') as outfile:
        pickle.dump(train_emb, outfile)
    with open(directory + args.dataset + '_' + args.method + '_' + args.strategy + '_' + str(args.window) + '_' + 'train_y.pkl', 'wb') as outfile:
        pickle.dump(d.train_labels, outfile)
    with open(directory + args.dataset + '_' + args.method + '_' + args.strategy + '_' + str(args.window) + '_' + 'test_x.pkl', 'wb') as outfile:
        pickle.dump(test_emb, outfile)
    with open(directory + args.dataset + '_' + args.method + '_' + args.strategy + '_' + str(args.window) + '_' + 'test_y.pkl', 'wb') as outfile:
        pickle.dump(d.test_labels, outfile)

if __name__ == "__main__":
    main()
