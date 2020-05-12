#! /usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def sentences_min(sentences):
    return len(min(sentences, key = lambda i: len(i)))

def sentences_max(sentences):
    return len(max(sentences, key = lambda i: len(i)))

def sentences_len(sentences):
    lengths = []
    for i in sentences:
        lengths.append(len(i))
    return np.array(lengths)

def sentences_measures(sentences):
    lengths = sentences_len(sentences)
    return(np.mean(lengths), np.median(lengths), np.std(lengths))

def discover_sentences_size(sentences):
    s_min = sentences_min(sentences) 
    s_max = sentences_max(sentences)
    print("min: ", s_min, "max: ", s_max)
    s_mean, s_median, s_std = sentences_measures(sentences)
    print("Mean len: ", s_mean, "Median len: ", s_median, "std: ", s_std)
    print("Quantas vezes a maior frase e maior que a media: ", s_max/s_mean)
    print("Quantas vezes a maior frase e maior que a mediana: ", s_max/s_median)
    print("Quantas vezes a maior frase e maior que o desvio padrao: ", s_max/s_std)
    print("Quantas vezes a maior frase e maior que a media + desvio padrao: ", s_max/(s_mean+s_std))

def sentences_percentile(all_sentences):
    lengths = sentences_len(all_sentences)
    percent = np.percentile(lengths, 90)
    print(percent)
    return percent

def sentences_histogram(all_sentences, dataset):
    lengths = sentences_len(all_sentences)
    plt.hist(lengths, bins=100)
    plt.title('Tamanho das sentenças: ' + dataset)
    plt.savefig("analysis/sentences/" + dataset + "_sentences_len_histogram.png")
    plt.close()
    sentences_boxplot(lengths, dataset)

def sentences_boxplot(lengths, dataset):
    fig1, ax1 = plt.subplots()
    ax1.set_title('Tamanho das sentenças: ' + dataset)
    ax1.boxplot(lengths)
    plt.savefig("analysis/sentences/" + dataset + "_sentences_len_boxplot.png")
    plt.close()
