#! /usr/bin/env python3

import argparse
from text_graph.text_graph import TextGraph
from text_graph.node_features import NodeFeatures

def main():
    g = TextGraph("Test")
    v = NodeFeatures()
    print("== MODEL LOADED ==")
    sentence = "cat dog tree woman human car queen king"
    l_sentence = sentence.split()
    for s in l_sentence:
        g.add_vertex(s)
    for s in range(0, len(l_sentence)-1):
        w1 = l_sentence[s]
        w2 = l_sentence[s+1]
        g.add_edge(w1, w2, v.edge_weight(w1, w2))
    g.plot_graph()

if __name__ == "__main__":
    main()
