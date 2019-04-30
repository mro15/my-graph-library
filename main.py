#! /usr/bin/env python3

import argparse
from text_graph.text_graph import TextGraph
from text_graph.node_features import NodeFeatures
from text_handler.dataset import Dataset
import utils

def main():
    d = Dataset("Polarity")
    d.read_polarity()
    d.pre_process_data()
    d.voc_stats()
    #for now I will not remove any word
    #d.remove_words()
    #d.voc_stats()

    train_graphs, test_graphs = utils.graph_strategy_one(d)
    print(len(train_graphs), len(test_graphs))

    #plot graphs
    #for i in train_graphs:
    #    i.plot_graph()
if __name__ == "__main__":
    main()
