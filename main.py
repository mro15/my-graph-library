#! /usr/bin/env python3

import argparse
from text_graph.text_graph import TextGraph
from text_graph.node_features import NodeFeatures
from text_handler.dataset import Dataset
import utils
import analysis.vocabulary as an
from representation_learning.representation_learning import RepresentationLearning
from features.features import Features

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--dataset', type=str, choices=["imdb", "polarity"], help='dataset name', required=True)   
    parser.add_argument('--method', type=str, choices=["node2vec", "bert"], help='representation method', required=True)
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
def plot_graphs(train_graphs, test_graphs):
    cont = 0
    for i in train_graphs:
        if cont < 10:
            i.plot_graph()
            cont = cont + 1
    cont = 0
    for i in test_graphs:
        if cont < 10:
            i.plot_graph()
            cont = cont + 1

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

    d.pre_process_data()
    #for now I will not remove any word
    #d.remove_words()

    #graph construction
    train_graphs, test_graphs = utils.graph_strategy_two(d, 3)
    print(len(train_graphs), len(test_graphs))

    #extract and write graph features
    features_out = Features(d.dataset)
    file_train = features_out.open_file("train")
    print("=== STARTING RL IN TRAIN GRAPHS ===")
    for i in range(0, len(train_graphs)):
        rl = RepresentationLearning(train_graphs[i], args.method)
        rl.initialize_rl_class()
        rl.representation_method.initialize_model()
        rl.representation_method.train()
        rl.set_features()
        feat = rl.get_features()
        features_out.write_in_file(file_train, feat, str(d.train_labels[i]))
    print("=== FINISHED RL IN TRAIN GRAPHS ===")

    file_test = features_out.open_file("test")
    print("=== STARTING RL IN TEST GRAPHS ===")
    for i in range(0, len(test_graphs)):
        rl = RepresentationLearning(test_graphs[i], args.method)
        rl.initialize_rl_class()
        rl.representation_method.initialize_model()
        rl.representation_method.train()
        rl.set_features()
        feat = rl.get_features()
        features_out.write_in_file(file_test, feat, str(d.test_labels[i]))
    print("=== FINISHED RL IN TEST GRAPHS ===")

if __name__ == "__main__":
    main()
