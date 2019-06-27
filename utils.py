#! /usr/bin/env python3

from text_graph.text_graph import TextGraph
from text_graph.node_features import NodeFeatures
from text_handler.dataset import Dataset
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import itertools
from collections import Counter
from math import log

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

#each node is a word from document and has no edge weight
def graph_strategy_two(d, k):
    train_graphs = []
    test_graphs = []
    print("BUILDING TRAIN GRAPHS")
    progress = tqdm(d.train_data)
    for i in progress:
        windows = []
        g = TextGraph(d.dataset)
        size = len(i)
        if size > k:
            windows += build_windows(i, k)
        else:
            windows.append(i)
        total_windows = len(windows)
        pair_windows = windows_in_pair(windows)
        for words, freq in pair_windows.items():
            w1, w2 = words
            g.add_vertex(w1)
            g.add_vertex(w2)
            g.add_edge(w1, w2)
        if len(list(pair_windows))<1:
            g.add_vertex(i[0])
        """
        #debug
        print("---- NODES ----")
        print(g.nodes())
        print("---- EDGES ----")
        print(g.edges())
        plot_graph(g.graph)
        exit()
        """
        train_graphs.append(g.graph)
    print("FINISHED TRAIN GRAPHS")

    print("BUILDING TEST GRAPHS")
    progress = tqdm(d.test_data)
    for i in progress:
        windows = []
        g = TextGraph(d.dataset)
        size = len(i)
        if size > k:
            windows += build_windows(i, k)
        else:
            windows.append(i)
        total_windows = len(windows)
        pair_windows = windows_in_pair(windows)
        for words, freq in pair_windows.items():
            w1, w2 = words
            g.add_vertex(w1)
            g.add_vertex(w2)
            g.add_edge(w1, w2)

        if len(list(pair_windows))<1:
            g.add_vertex(i[0])
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

def windows_in_pair(windows):
    windows_in_pair = Counter()
    for i in windows:
        pairs = list(itertools.combinations(i, 2))
        windows_in_pair.update(pairs)
    return windows_in_pair


def windows_in_word(windows):
    windows_in_word = Counter()
    for i in windows:
        windows_in_word.update(i)
    return windows_in_word

def build_windows(text, k):
    iterable = iter(text)
    result = tuple(itertools.islice(iterable, k))
    if len(result)==k:
        yield list(result)
    for element in iterable:
        result = result[1:] + (element,)
        yield list(result)

#each node is a word from document and edge weight is given by PMI
def graph_strategy_three(d, k):
    train_graphs = []
    test_graphs = []
    print("BUILDING TRAIN GRAPHS")
    progress = tqdm(d.train_data)
    for i in progress:
        windows = []
        g = TextGraph(d.dataset)
        size = len(i)
        if size > k:
            windows += build_windows(i, k)
        else:
            windows.append(i)
        total_windows = len(windows)
        pair_windows = windows_in_pair(windows)
        word_windows = windows_in_word(windows)
        for words, freq in pair_windows.items():
            w1, w2 = words
            pmi = log((freq/total_windows)/((word_windows[w1]*word_windows[w2])/(total_windows*total_windows)))
            if pmi >= 0:
                g.add_vertex(w1)
                g.add_vertex(w2)
                g.add_weight_edge(w1, w2, pmi)
        if ((len(list(pair_windows))<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        """
        #debug
        print("---- NODES ----")
        print(g.nodes())
        print("---- EDGES ----")
        print(g.edges())
        plot_graph(g.graph)
        exit()
        """
        train_graphs.append(g.graph)
    print("FINISHED TRAIN GRAPHS")

    print("BUILDING TEST GRAPHS")
    progress = tqdm(d.test_data)
    for i in progress:
        windows = []
        g = TextGraph(d.dataset)
        size = len(i)
        if size > k:
            windows += build_windows(i, k)
        else:
            windows.append(i)
        total_windows = len(windows)
        pair_windows = windows_in_pair(windows)
        word_windows = windows_in_word(windows)
        for words, freq in pair_windows.items():
            w1, w2 = words
            pmi = log((freq/total_windows)/((word_windows[w1]*word_windows[w2])/(total_windows*total_windows)))
            if pmi >=  0:
                g.add_vertex(w1)
                g.add_vertex(w2)
                g.add_weight_edge(w1, w2, pmi)
        if ((len(list(pair_windows))<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])

        test_graphs.append(g.graph) 
    print("FINISHED TEST GRAPHS") 

    return train_graphs, test_graphs

