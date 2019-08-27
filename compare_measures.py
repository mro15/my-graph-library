#! /usr/bin/env python3

import argparse
import utils
from text_graph.text_graph import TextGraph
from text_graph.node_features import NodeFeatures
from text_handler.dataset import Dataset

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--dataset', type=str, choices=["imdb", "polarity", "mr"], help='dataset name', required=True)   
    parser.add_argument('--window', type=int,  help='window size', required=True)
    return parser.parse_args()



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
    #all_edges_train, all_edges_test = utils.graph_strategy_two(d, args.window)
    pmi_1990_edges_train, pmi_1990_edges_test = utils.graph_strategy_five(d, args.window)
    #for i in range(0,1000):
        #print(all_edges_train[i].number_of_nodes(), all_edges_train[i].number_of_edges(), " --- ", pmi_1990_edges_train[i].number_of_nodes(), pmi_1990_edges_train[i].number_of_edges())



if __name__ == "__main__":
    main()
