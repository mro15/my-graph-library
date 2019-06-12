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

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--dataset', type=str, choices=["imdb", "polarity"], help='dataset name', required=True)   
    parser.add_argument('--method', type=str, choices=["node2vec", "gcn"], help='representation method', required=True)
    parser.add_argument('--strategy', type=str, choices=["no_weight", "pmi"], help='representation method', required=True)
    parser.add_argument('--window', type=int,  help='window size', required=True)
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

def graph_methods(d, method, window_size, strategy):
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
    elif strategy == "pmi":
        train_graphs, test_graphs = utils.graph_strategy_three(d, window_size)
        weight = True
    else:
        exit()

    #extract and write ddgraph features
    features_out = Features(d.dataset)
    file_train = features_out.open_file(method + "train")
    print("=== STARTING RL IN TRAIN GRAPHS ===")
    for i in range(0, len(train_graphs)):
        rl = RepresentationLearning(train_graphs[i], method, weight, d.train_data[i])
        rl.initialize_rl_class()
        rl.representation_method.initialize_model()
        rl.representation_method.train()
        rl.set_features()
        feat = rl.get_features()
        features_out.write_in_file(file_train, feat, str(d.train_labels[i]))
    print("=== FINISHED RL IN TRAIN GRAPHS ===")

    file_test = features_out.open_file(method + "test")
    print("=== STARTING RL IN TEST GRAPHS ===")
    for i in range(0, len(test_graphs)):
        rl = RepresentationLearning(test_graphs[i], method, weight, d.test_data[i])
        rl.initialize_rl_class()
        rl.representation_method.initialize_model()
        rl.representation_method.train()
        rl.set_features()
        feat = rl.get_features()
        features_out.write_in_file(file_test, feat, str(d.test_labels[i]))
    print("=== FINISHED RL IN TEST GRAPHS ===")

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
    else:
        print("Error: dataset name unknown")
        return 1

    g_methods = ["node2vec"]
    if args.method in g_methods:
        graph_methods(d, args.method, args.window, args.strategy)
    else:
        vector_methods(d, args.method)

if __name__ == "__main__":
    main()
