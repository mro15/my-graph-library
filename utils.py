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

def word_count(sentence):
    return Counter(sentence)

def build_windows(text, k):
    iterable = iter(text)
    result = tuple(itertools.islice(iterable, k))
    if len(result)==k:
        yield list(result)
    for element in iterable:
        result = result[1:] + (element,)
        yield list(result)

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


#each node is a word from document and edge weight is given by PMI(yao, 2019)
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

#each node is a word from document and edge weight is given by PMI(yao, 2019) normalized
#with min max norm
def graph_strategy_norm_two(d, k):
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
        #calculate pmi for all edges of the document
        pmi_vet = []
        for words, freq in pair_windows.items():
            w1, w2 = words
            pmi = log((freq/total_windows)/((word_windows[w1]*word_windows[w2])/(total_windows*total_windows)))
            pmi_vet.append(pmi)
        pmi_vet = np.array(pmi_vet).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(pmi_vet)
        pmi_vet_n = scaler.transform(pmi_vet)
        if len(pmi_vet_n) == len(list(pair_windows)):
            pmi_pos = 0
            for words, freq in pair_windows.items():
                w1, w2 = words
                g.add_vertex(w1)
                g.add_vertex(w2)
                g.add_weight_edge(w1, w2, (pmi_vet_n[pmi_pos][0]*100))
                pmi_pos = pmi_pos + 1
        else:
            print(np.shape(pmi_vet_n), len(list(pair_windows)), len(pmi_vet_n))
            print("something wrong in normalized pmi vector in train graphs")
            exit()
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
        #calculate pmi for all edges of the document
        pmi_vet = []
        for words, freq in pair_windows.items():
            w1, w2 = words
            pmi = log((freq/total_windows)/((word_windows[w1]*word_windows[w2])/(total_windows*total_windows)))
            pmi_vet.append(pmi)
        pmi_vet = np.array(pmi_vet).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(pmi_vet)
        pmi_vet_n = scaler.transform(pmi_vet)
        if len(pmi_vet_n) == len(list(pair_windows)):
            pmi_pos = 0
            for words, freq in pair_windows.items():
                w1, w2 = words
                g.add_vertex(w1)
                g.add_vertex(w2)
                g.add_weight_edge(w1, w2, (pmi_vet_n[pmi_pos][0]*100))
                pmi_pos = pmi_pos + 1
        else:
            print(np.shape(pmi_vet_n), len(list(pair_windows)), len(pmi_vet_n))
            print("something wrong in normalized omi vector in train graphs")
            exit()
        if ((len(list(pair_windows))<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])

        test_graphs.append(g.graph) 
    print("FINISHED TEST GRAPHS") 

    return train_graphs, test_graphs

def graph_strategy_three_all(d, k):
    train_graphs = []
    test_graphs = []
    windows = bcf.from_words(all_docs_to_one_tokens_list(d), window_size=k)
    llr_all = dict(windows.score_ngrams(bam.pmi))
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.pmi):
                llr = llr_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if llr >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, llr)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        train_graphs.append(g.graph)
        """
        #debug
        print("---- NODES ----")
        print(g.nodes())
        print("---- EDGES ----")
        print(g.edges())
        plot_graph(g.graph)
        break
        """
    print("FINISHED GRAPHS FROM TRAIN DATASET")
    print("BUILDING GRAPHS FROM TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.pmi):
                #print(pairs)
                #print(llr_all[pairs[0]])
                llr = llr_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if llr >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, llr)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
        """
        #debug
        print("---- NODES ----")
        print(g.nodes())
        print("---- EDGES ----")
        print(g.edges())
        plot_graph(g.graph)
        break
        """
    print("FINISHED GRAPHS FROM TEST DATASET")
    return train_graphs, test_graphs

#word association measure defined as PMI (Pointwise Mutual Information)
#by  Church and Hankis 1990.
def graph_strategy_three(d, k):
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
                if pmi >= 0:
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
                if pmi >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, pmi)
        if((len(pairs)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

def graph_strategy_four_all(d, k):
    train_graphs = []
    test_graphs = []
    windows = bcf.from_words(all_docs_to_one_tokens_list(d), window_size=k)
    llr_all = dict(windows.score_ngrams(bam.dice))
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.dice):
                llr = llr_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if llr >= 0:
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
            for pairs in t_windows.score_ngrams(bam.dice):
                #print(pairs)
                #print(llr_all[pairs[0]])
                llr = llr_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if llr >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, llr)
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
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

def graph_strategy_five_all(d, k):
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
                #print(pairs)
                #print(llr_all[pairs[0]])
                llr = llr_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if llr >= 0:
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
def graph_strategy_five(d, k):
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
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

def graph_strategy_six_all(d, k):
    train_graphs = []
    test_graphs = []
    windows = bcf.from_words(all_docs_to_one_tokens_list(d), window_size=k)
    llr_all = dict(windows.score_ngrams(bam.chi_sq))
    print("BUILDING GRAPHS FROM TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.chi_sq):
                llr = llr_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if llr >= 0:
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
            for pairs in t_windows.score_ngrams(bam.chi_sq):
                #print(pairs)
                #print(llr_all[pairs[0]])
                llr = llr_all[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                if llr >= 0:
                    g.add_vertex(w1)
                    g.add_vertex(w2)
                    g.add_weight_edge(w1, w2, llr)
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

#word association measure defined as Chi-Square
def graph_strategy_six(d, k):
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
        if((len(i)<1) or (len(g.nodes())==0)):
            g.add_vertex(i[0])
        test_graphs.append(g.graph)
    print("FINISHED GRAPHS FROM TEST DATASET")

    return train_graphs, test_graphs

