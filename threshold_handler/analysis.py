#! /usr/bin/env python3

from text_graph.text_graph import TextGraph
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
import utils


def get_bam_function(strategy):
    functions = {
        "pmi": bam.pmi,
        "llr": bam.likelihood_ratio, 
        "dice": bam.dice,
        "chi_square": bam.chi_sq
    }
    return functions[strategy]

def histogram_strategy_local(d, k, strategy, output_name):
    print("CALCULATING MEASURE FOR TRAIN DATASET")
    values = []
    progress = tqdm(d.train_data)
    for i in progress:
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.score_ngrams(get_bam_function(strategy)):
                pmi = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                values.append(pmi)
    
    print("CALCULATING MEASURE FOR TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.score_ngrams(get_bam_function(strategy)):
                pmi = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                values.append(pmi)
    print(len(values))

    plt.hist(values)
    plt.savefig(output_name)
    print("save")
    plt.close()

def histogram_strategy_global(d, k, strategy, output_name):
    values = []
    windows = bcf.from_words(utils.all_docs_to_one_tokens_list(d), window_size=k)
    global_pmi = dict(windows.score_ngrams(get_bam_function(strategy)))
    print("CALCULATING MEASURE FOR TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(get_bam_function(strategy)):
                pmi = global_pmi[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                values.append(pmi)

    print("CALCULATING MEASURE FOR TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.pmi):
                pmi = global_pmi[pairs[0]]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                values.append(pmi)
    print(len(values))

    plt.hist(values)
    plt.savefig(output_name)
    print("save global")
    plt.close()

