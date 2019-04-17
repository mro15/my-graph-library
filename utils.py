#! /usr/bin/env python3

from text_graph.text_graph import TextGraph
from text_graph.node_features import NodeFeatures
from text_handler.dataset import Dataset

def graph_strategy_one(d):
    train_graphs = []
    for i in range(0, len(d.train_data)):
        g = TextGraph("Polarity")
        for word in d.vocabulary:
            g.add_vertex(word)
        #word co-occurrence size 2
        for s in range(0, len(d.train_data[i])-1):
                w1 = d.train_data[i][s]
                w2 = d.train_data[i][s+1]
                g.add_edge(w1, w2, 0)
        train_graphs.append(g)
    
    test_graphs = []
    for i in range(0, len(d.test_data)):
        g = TextGraph("Polarity")
        for word in d.vocabulary:
            g.add_vertex(word)
        #word co-occurrence size 2
        for s in range(0, len(d.test_data[i])-1):
                w1 = d.test_data[i][s]
                w2 = d.test_data[i][s+1]
                g.add_edge(w1, w2, 0)
        test_graphs.append(g)
    
    return train_graphs, test_graphs
