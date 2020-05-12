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

# TODO: update
def histogram_freq_local(d, k, output_name):
    print("CALCULATING FREQ FOR TRAIN DATASET")
    values = []
    progress = tqdm(d.train_data)
    for i in progress:
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.ngram_fd.items():
                pmi = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                values.append(pmi)

    print("CALCULATING FREQ FOR TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.ngram_fd.items():
                pmi = pairs[1]
                w1 = pairs[0][0]
                w2 = pairs[0][1]
                values.append(pmi)
    print(len(values))

    fig, axs = plt.subplots(1, 2)
    axs[0].hist(values)
    axs[1].boxplot(values)
    plt.savefig(output_name)
    plt.close()

# TODO: update
def histogram_freq_global(d, k, output_name):
    values = []
    windows = bcf.from_words(utils.all_docs_to_one_tokens_list(d), window_size=k)
    global_pmi = dict(windows.ngram_fd.items())
    print("CALCULATING MEASURE FOR TRAIN DATASET")
    progress = tqdm(d.train_data)
    for i in progress:
        g = TextGraph(d.dataset)
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.pmi):
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

    fig, axs = plt.subplots(1, 2)
    axs[0].hist(values)
    axs[1].boxplot(values)
    plt.savefig(output_name)
    plt.close()


def histogram_strategy_local(d, k, strategy, output_name):
    print("CALCULATING MEASURE FOR TRAIN DATASET")
    values = []
    progress = tqdm(d.train_data)
    for i in progress:
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.score_ngrams(get_bam_function(strategy)):
                pmi = pairs[1]
                values.append(pmi)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                try:
                    for pairs in windows.score_ngrams(get_bam_function(strategy)):
                        pmi = pairs[1]
                        values.append(pmi)
                except:
                    i_unique = list(set(i))
                    windows = bcf.from_words(i_unique, window_size=len(i_unique))
                    for pairs in windows.score_ngrams(get_bam_function(strategy)):
                        pmi = pairs[1]
                        values.append(pmi)
                    print(i_unique, windows, pairs)

    print("CALCULATING MEASURE FOR TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        if len(i) > k:
            windows = bcf.from_words(i, window_size=k)
            for pairs in windows.score_ngrams(get_bam_function(strategy)):
                pmi = pairs[1]
                values.append(pmi)
        else:
            if len(i) > 1:
                windows = bcf.from_words(i, window_size=len(i))
                try:
                    for pairs in windows.score_ngrams(get_bam_function(strategy)):
                        pmi = pairs[1]
                        values.append(pmi)
                except:
                    i_unique = list(set(i))
                    windows = bcf.from_words(i_unique, window_size=len(i_unique))
                    for pairs in windows.score_ngrams(get_bam_function(strategy)):
                        pmi = pairs[1]
                        values.append(pmi)
                    print(i_unique, windows, pairs)

    print(len(values))

    fig, axs = plt.subplots(1, 2)
    percent = np.percentile(values, 90)
    values_zoom = [v for v in values if v <= percent]
    axs[0].hist(values_zoom)
    axs[1].boxplot(values)
    plt.savefig(output_name)
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
            for pairs in t_windows.score_ngrams(bam.pmi):
                pmi = global_pmi[pairs[0]]
                values.append(pmi)
        else:
            if len(i) > 1:
                t_windows = bcf.from_words(i, window_size=len(i))
                try:
                    for pairs in t_windows.score_ngrams(bam.pmi):
                        pmi = global_pmi[pairs[0]]
                        values.append(pmi)
                except:
                    i_unique = list(set(i))
                    t_windows = bcf.from_words(i_unique, window_size=len(i_unique))
                    for pairs in t_windows.score_ngrams(get_bam_function(strategy)):
                        pmi = global_pmi[pairs[0]]
                        values.append(pmi)
                    print(i_unique, windows, pairs)

    print("CALCULATING MEASURE FOR TEST DATASET")
    progress = tqdm(d.test_data)
    for i in progress:
        if len(i) > k:
            t_windows = bcf.from_words(i, window_size=k)
            for pairs in t_windows.score_ngrams(bam.pmi):
                pmi = global_pmi[pairs[0]]
                values.append(pmi)
        else:
            if len(i) > 1:
                t_windows = bcf.from_words(i, window_size=len(i))
                try:
                    for pairs in t_windows.score_ngrams(bam.pmi):
                        pmi = global_pmi[pairs[0]]
                        values.append(pmi)
                except:
                    i_unique = list(set(i))
                    t_windows = bcf.from_words(i_unique, window_size=len(i_unique))
                    for pairs in t_windows.score_ngrams(get_bam_function(strategy)):
                        pmi = global_pmi[pairs[0]]
                        values.append(pmi)
                    print(i_unique, windows, pairs)

    print(len(values))
    fig, axs = plt.subplots(1, 2)
    percent = np.percentile(values, 90)
    values_zoom = [v for v in values if v <= percent]
    axs[0].hist(values_zoom)
    axs[1].boxplot(values)
    plt.savefig(output_name)
    plt.close()
