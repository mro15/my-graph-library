#! /usr/bin/env python3

from text_graph.text_graph import TextGraph
from text_graph.node_features import NodeFeatures
from text_handler.dataset import Dataset
import matplotlib.pyplot as plt
import networkx as nx

"""
    parameters:
        edge size
        graph strategy
        number of nodes
"""


#each node is a word from vocabulary
def graph_strategy_one(d):
    train_graphs = []
    for i in range(0, len(d.train_data)):
        g = TextGraph(d.dataset)
        for word in d.vocabulary:
            g.add_vertex(word)
        #word co-occurrence size 2
        for s in range(0, len(d.train_data[i])-1):
                w1 = d.train_data[i][s]
                w2 = d.train_data[i][s+1]
                g.add_edge(w1, w2)
        #convert graph to sparse matrix

        #train_graphs.append(g)
    
    print("FINISHED TRAIN GRAPHS") 

    test_graphs = []
    for i in range(0, len(d.test_data)):
        g = TextGraph(d.dataset)
        for word in d.vocabulary:
            g.add_vertex(word)
        #word co-occurrence size 2
        for s in range(0, len(d.test_data[i])-1):
                w1 = d.test_data[i][s]
                w2 = d.test_data[i][s+1]
                g.add_edge(w1, w2)
        #test_graphs.append(g)
    
    print("FINISHED TEST GRAPHS") 
    return train_graphs, test_graphs

#each node is a word from document
def graph_strategy_two(d, k):
    train_graphs = []
    test_graphs = []
    #build nodes
    #pass coocurrence window to build edges
    #build train graphs
    for i in d.train_data:
        g = TextGraph(d.dataset)
        #print("sentence: ", i)
        for j in range(0, len(i)-k):
            w1 = i[j]
            g.add_vertex(w1)
            for s in range(j+1, j+k):
                w2 = i[s]
                g.add_vertex(w2)
                g.add_edge(w1, w2)
        #remainder
        for r in range(j, len(i)):
            w1 = i[r]
            for rn in range(r+1, len(i)):
                w2 = i[rn]
                g.add_vertex(w2)
                g.add_edge(w1, w2)
        """
        #debug
        print("---- NODES ----")
        print(g.nodes())
        print("---- EDGES ----")
        print(g.edges())
        g.plot_graph()
        exit()
        """
        train_graphs.append(g.graph)

    print("FINISHED TRAIN GRAPHS") 
    for i in d.test_data:
        g = TextGraph(d.dataset)
        for j in range(0, len(i)-k):
            w1 = i[j]
            g.add_vertex(w1)
            for s in range(j+1, j+k):
                w2 = i[s]
                g.add_vertex(w2)
                g.add_edge(w1, w2)
        #remainder
        for r in range(j, len(i)):
            w1 = i[r]
            for rn in range(r+1, len(i)):
                w2 = i[rn]
                g.add_vertex(w2)
                g.add_edge(w1, w2)
        test_graphs.append(g.graph)
    
    print("FINISHED TEST GRAPHS") 

    return train_graphs, test_graphs

def plot_graph(g):
        options = {'node_color': 'lightskyblue', 'node_size': 5000, 'with_labels': 'True'}
        edge_labels = nx.get_edge_attributes(g,'weight')
        pos=nx.spring_layout(g)
        nx.draw(g, pos, **options)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
        plt.show()

def my_pmi(d):
    windows = {}
    return windows
