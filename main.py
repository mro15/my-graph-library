#! /usr/bin/env python3

import argparse
from text_graph.text_graph import TextGraph
from text_graph.node_features import NodeFeatures
from text_handler.dataset import Dataset
import utils
import analysis.vocabulary as an

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--dataset', type=str, help='dataset name [polarity, imdb]', required=True)
    return parser.parse_args()

def main():
    args = read_args()
    if args.dataset == "polarity":
        d = Dataset(args.dataset)
        d.read_polarity()
    elif args.dataset == "imdb":
        d = Dataset(args.dataset)
        d.read_imdb()
        d.small_debug()
    else:
        print("Error: dataset name unknown")
        return 1

    d.pre_process_data()
    #for now I will not remove any word
    #d.remove_words()

    an.all_voc_analysis(d)
    pos_voc = an.pos_voc_analysis(d)
    neg_voc = an.neg_voc_analysis(d)
    an.plot_analysis(d.vocabulary, "words", "count", args.dataset + " all classes", args.dataset + "_all")
    an.plot_analysis(pos_voc, "words", "count", args.dataset + " positive class", args.dataset + "_pos")
    an.plot_analysis(neg_voc, "words", "count", args.dataset + " negative class", args.dataset + "_neg")

    #train_graphs, test_graphs = utils.graph_strategy_one(d)
    #print(len(train_graphs), len(test_graphs))
    #plot graphs
    #for i in train_graphs:
    #    i.plot_graph()

if __name__ == "__main__":
    main()
