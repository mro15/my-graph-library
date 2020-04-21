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
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from nltk.collocations import BigramAssocMeasures as bam
from nltk.collocations import BigramCollocationFinder as bcf

def plot_graph(g):
    options = {'node_color': 'lightskyblue', 'node_size': 5000, 'with_labels': 'True'}
    edge_labels = nx.get_edge_attributes(g,'weight')
    pos=nx.spring_layout(g)
    nx.draw(g, pos, **options)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    plt.show()

# todo: see a way of not do this
def all_docs_to_one_tokens_list(d):
    docs = d.train_data+d.test_data
    #docs = d.train_data[:2]
    token_list = list(itertools.chain(*docs))
    return token_list

#each node is a word from document and has no edge weight
def graph_strategy_one(d, k):
    train_graphs = []
    test_graphs = []
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.ngram_fd.items():
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                g.add_vertex(w1)
                g.add_vertex(w2)
                g.add_edge(w1, w2)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                for pairs in windows.ngram_fd.items():
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_edge(w1, w2)
        if((len(pairs)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        train_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TRAIN DATASET")

    print("BUILDING GRAPHS FROM TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.ngram_fd.items():
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                g.add_vertex(w1)
                g.add_vertex(w2)
                g.add_edge(w1, w2)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                for pairs in windows.ngram_fd.items():
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_edge(w1, w2)
        if((len(pairs)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

# frequency calculated over all documents from dataset
def graph_strategy_two_all(d, k, threshold=0):
    train_graphs = []
    test_graphs = []
    windows = bcf.from_words(all_docs_to_one_tokens_list(d), window_size=k)
    freq_all = dict(windows.ngram_fd.items())
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.pmi):
                freq = freq_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if freq > 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, freq)
        else:
            if len(i) > 1:
                t_windows = bcf.from_words(i, window_size=len(i))
                for pairs in t_windows.score_ngrams(bam.pmi):
                    freq = freq_all[pairs[0]]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if freq > threshold:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, freq)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        train_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TRAIN DATASET")

    print("BUILDING GRAPHS FROM TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.pmi):
                freq = freq_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if freq > 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, freq)
        else:
            if len(i) > 1 :
                t_windows = bcf.from_words(i, window_size=len(i))
                for pairs in t_windows.score_ngrams(bam.pmi):
                    freq = freq_all[pairs[0]]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if freq > threshold:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, freq)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")
    return train_graphs, test_graphs


# weight is the frequency of co-occurrence calculated over a single document
# for once
def graph_strategy_two(d, k, threshold=0):
    train_graphs = []
    test_graphs = []
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.ngram_fd.items():
                freq = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if freq > threshold:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, freq)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                for pairs in windows.ngram_fd.items():
                    freq = pairs[1]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if freq > threshold:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, freq)
        if((len(pairs)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        train_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TRAIN DATASET")

    print("BUILDING GRAPHS FROM TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.ngram_fd.items():
                freq = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if freq > threshold:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, freq)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                for pairs in windows.ngram_fd.items():
                    freq = pairs[1]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if freq > threshold:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, freq)
        if((len(pairs)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

# pmi 1990 calculated over all documents from dataset   
def graph_strategy_three_all(d, k, threshold=0):
    train_graphs = []
    test_graphs = []
    windows = bcf.from_words(all_docs_to_one_tokens_list(d), window_size=k)
    pmi_all = dict(windows.score_ngrams(bam.pmi))
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.pmi):
                pmi = pmi_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if pmi > 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, pmi)
        else:
            if len(i) > 1:
                t_windows = bcf.from_words(i, window_size=len(i))
                for pairs in t_windows.score_ngrams(bam.pmi):
                    pmi = pmi_all[pairs[0]]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if pmi > 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, pmi)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        train_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TRAIN DATASET")

    print("BUILDING GRAPHS FROM TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.pmi):
                pmi = pmi_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if pmi > 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, pmi)
        else:
            if len(i) > 1:
                t_windows = bcf.from_words(i, window_size=len(i))
                for pairs in t_windows.score_ngrams(bam.pmi):
                    pmi = pmi_all[pairs[0]]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if pmi > 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, pmi)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")
    return train_graphs, test_graphs

# word association measure defined as PMI (Pointwise Mutual Information)
# by  Church and Hankis 1990. Calculated over a single document from dataset
# for once
def graph_strategy_three(d, k, threshold=0):
    train_graphs = []
    test_graphs = []
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.score_ngrams(bam.pmi):
                pmi = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if pmi > threshold:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, pmi)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                for pairs in windows.score_ngrams(bam.pmi):
                    pmi = pairs[1]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if pmi > threshold:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, pmi)
        if((len(pairs)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        train_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TRAIN DATASET")
    
    print("BUILDING GRAPHS FROM TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.score_ngrams(bam.pmi):
                pmi = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if pmi > threshold:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, pmi)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                for pairs in windows.score_ngrams(bam.pmi):
                    pmi = pairs[1]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if pmi > threshold:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, pmi)

        if((len(pairs)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

#Dice(1945) calculated over all dataset
def graph_strategy_four_all(d, k, threshold=0):
    train_graphs = []
    test_graphs = []
    windows = bcf.from_words(all_docs_to_one_tokens_list(d), window_size=k)
    dice_all = dict(windows.score_ngrams(bam.dice))
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.dice):
                dice = dice_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if dice >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, dice)
        else:
            if len(i) > 1:
                t_windows = bcf.from_words(i, window_size=len(i))
                for pairs in t_windows.score_ngrams(bam.dice):
                    dice = dice_all[pairs[0]]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if dice >= 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, dice)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        train_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TRAIN DATASET")

    print("BUILDING GRAPHS FROM TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.dice):
                dice = dice_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if dice >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, dice)
        else:
            if len(i) > 1:
                t_windows = bcf.from_words(i, window_size=len(i))
                for pairs in t_windows.score_ngrams(bam.dice):
                    dice = dice_all[pairs[0]]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if dice >= 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, dice)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

#Dice(1945)
def graph_strategy_four(d, k):
    train_graphs = []
    test_graphs = []
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.score_ngrams(bam.dice):
                dice = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if dice >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, dice)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                for pairs in windows.score_ngrams(bam.dice):
                    dice = pairs[1]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if dice >= 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, dice)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        train_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TRAIN DATASET")
    
    print("BUILDING GRAPHS FROM TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.score_ngrams(bam.dice):
                dice = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if dice >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, dice)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                for pairs in windows.score_ngrams(bam.dice):
                    dice = pairs[1]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if dice >= 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, dice)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

# llr: log likelihood ratio
def graph_strategy_five_all(d, k, threshold=0):
    train_graphs = []
    test_graphs = []
    windows = bcf.from_words(all_docs_to_one_tokens_list(d), window_size=k)
    llr_all = dict(windows.score_ngrams(bam.likelihood_ratio))
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.likelihood_ratio):
                llr = llr_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if llr >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, llr)
        else:
            if len(i) > 1:
                t_windows = bcf.from_words(i, window_size=len(i))
                try:
                    for pairs in t_windows.score_ngrams(bam.likelihood_ratio):
                        llr = llr_all[pairs[0]]
                        w1 = pairs[0][0]
                        w2 = pairs[0][1]
                        if llr >= 0:
                            g.add_vertex(w1)
                            g.add_vertex(w2)
                            g.add_weight_edge(w1, w2, llr)
                except ValueError:
                    for words, v in t_windows.ngram_fd.items():
                        w1 = words[0]
                        w2 = words[1]
                        llr = 1
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, llr)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        train_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TRAIN DATASET")

    print("BUILDING GRAPHS FROM TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.likelihood_ratio):
                llr = llr_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if llr >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, llr)
        else:
            if len(i) > 1:
                t_windows = bcf.from_words(i, window_size=len(i))
                try:
                    for pairs in t_windows.score_ngrams(bam.likelihood_ratio):
                        llr = llr_all[pairs[0]]
                        w1 = pairs[0][0]
                        w2 = pairs[0][1]
                        if llr >= 0:
                            g.add_vertex(w1)
                            g.add_vertex(w2)
                            g.add_weight_edge(w1, w2, llr)
                except ValueError:
                    for words, v in t_windows.ngram_fd.items():
                        w1 = words[0]
                        w2 = words[1]
                        llr = 1
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, llr)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

#word association measure defined as LLR (Log Likelihood Ratio)
#by Dunning 1993
def graph_strategy_five(d, k, threshold=0):
    train_graphs = []
    test_graphs = []
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.score_ngrams(bam.likelihood_ratio):
                llr = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if llr >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, llr)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                try:
                    for pairs in windows.score_ngrams(bam.likelihood_ratio):
                        llr = pairs[1]
                        w1 = pairs[0][0]
                        w2 = pairs[0][1]
                        if llr >= 0:
                            g.add_vertex(w1)
                            g.add_vertex(w2)
                            g.add_weight_edge(w1, w2, llr)
                except ValueError:
                    for words, v in windows.ngram_fd.items():
                        w1 = words[0]
                        w2 = words[1]
                        llr = 1
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, llr)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        train_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TRAIN DATASET")
    
    print("BUILDING GRAPHS FROM TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.score_ngrams(bam.likelihood_ratio):
                llr = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if llr >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, llr)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                try:
                    for pairs in windows.score_ngrams(bam.likelihood_ratio):
                        llr = pairs[1]
                        w1 = pairs[0][0]
                        w2 = pairs[0][1]
                        if llr >= 0:
                            g.add_vertex(w1)
                            g.add_vertex(w2)
                            g.add_weight_edge(w1, w2, llr)
                except ValueError:
                    for words, v in windows.ngram_fd.items():
                        w1 = words[0]
                        w2 = words[1]
                        llr = 1
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, llr)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

# chi square calculated over all dataset documents
def graph_strategy_six_all(d, k, threshold=0):
    train_graphs = []
    test_graphs = []
    windows = bcf.from_words(all_docs_to_one_tokens_list(d), window_size=k)
    chi_all = dict(windows.score_ngrams(bam.chi_sq))
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.chi_sq):
                chi = chi_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if chi >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, chi)
        else:
            if len(i) > 1:
                t_windows = bcf.from_words(i, window_size=len(i))
                for pairs in t_windows.score_ngrams(bam.chi_sq):
                    chi = chi_all[pairs[0]]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if chi >= 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, chi)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        train_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TRAIN DATASET")

    print("BUILDING GRAPHS FROM TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.chi_sq):
                chi = chi_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if chi >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, chi)
        else:
            if len(i) > 1:
                t_windows = bcf.from_words(i, window_size=len(i))
                for pairs in t_windows.score_ngrams(bam.chi_sq):
                    chi = chi_all[pairs[0]]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if chi >= 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, chi)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

#word association measure defined as Chi-Square
def graph_strategy_six(d, k, threshold=0):
    train_graphs = []
    test_graphs = []
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.score_ngrams(bam.chi_sq):
                chi = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if chi >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, chi)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                for pairs in windows.score_ngrams(bam.chi_sq):
                    chi = pairs[1]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if chi >= 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, chi)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        train_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TRAIN DATASET")
    
    print("BUILDING GRAPHS FROM TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.score_ngrams(bam.chi_sq):
                chi = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if chi >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, chi)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                for pairs in windows.score_ngrams(bam.chi_sq):
                    chi = pairs[1]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if chi >= 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, chi)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs
