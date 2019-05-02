#! /usr/bin/env python3

from text_handler.dataset import Dataset
import collections
import matplotlib.pyplot as plt
import numpy as np

def all_voc_analysis(d):
    print("VOCABULARY SIZE: ", len(d.vocabulary))
    print("=== MOST COMMON WORDS ===")
    print(d.vocabulary.most_common(100))
    print("=== MOST UNCOMMON WORDS ===")
    print(d.vocabulary.most_common()[:-100-1:-1])

def pos_voc_analysis(d):
    pos_voc = collections.Counter()
    indexes = [i for i, x in enumerate(d.train_labels) if x == 1]
    pos_sentences_train = [d.train_data[x] for x in indexes]
    print(len(pos_sentences_train))
    indexes = [i for i, x in enumerate(d.test_labels) if x == 1]
    pos_sentences_test = [d.test_data[x] for x in indexes]
    print(len(pos_sentences_test))
    for tokens in pos_sentences_train + pos_sentences_test:
        pos_voc.update(tokens)

    print("====== Positive Vocabulary ======")
    print("Size: ", len(pos_voc))
    print("Most common words")
    print(pos_voc.most_common(100))
    print("=== MOST UNCOMMON WORDS ===")
    print(pos_voc.most_common()[:-100-1:-1])
    return pos_voc

def neg_voc_analysis(d):
    neg_voc = collections.Counter()
    indexes = [i for i, x in enumerate(d.train_labels) if x == 0]
    neg_sentences_train = [d.train_data[x] for x in indexes]
    print(len(neg_sentences_train))
    indexes = [i for i, x in enumerate(d.test_labels) if x == 0]
    neg_sentences_test = [d.test_data[x] for x in indexes]
    print(len(neg_sentences_test))
    for tokens in neg_sentences_train + neg_sentences_test:
        neg_voc.update(tokens)

    print("====== Negative Vocabulary ======")
    print("Size: ", len(neg_voc))
    print("Most common words")
    print(neg_voc.most_common(100))
    print("=== MOST UNCOMMON WORDS ===")
    print(neg_voc.most_common()[:-100-1:-1])
    return neg_voc

def plot_analysis(vocabulary, x_name, y_name, g_name, g_save):
    d = collections.OrderedDict(vocabulary.most_common(100))
    x = list(range(100))
    words = list(d.keys())
    counts = list(d.values())
    #print(words, counts)
    plt.figure(figsize=(15,5))
    plt.bar(x, counts)
    plt.xticks(x, words)
    plt.xticks(rotation=90)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(g_name)
    plt.savefig("analysis/" + g_save + ".png", bbox_inches = "tight")
    plt.close()
