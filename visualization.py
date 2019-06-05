#! /usr/bin/env python3

import argparse
from features.features import Features
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--view', type=str, choices=["tsne", "pca"], help='visualization method', required=True)   
    parser.add_argument('--train', type=str, help='path for train file', required=True)   
    parser.add_argument('--test', type=str, help='path for test file', required=True)   
    parser.add_argument('--dataset', type=str, help='path for test file', required=True)   
    return parser.parse_args()

def read_features(train_path, test_path, i_x, i_y):
    f = open(train_path, "r")
    x_train = []
    y_train = []
    for line in f:
        splited = line.split()
        y_train.append(int(splited[0]))
        x_train.append([float(x) for x in splited[i_x:i_y]])
    f.close()

    f = open(test_path, "r")
    x_test = []
    y_test = []
    for line in f:
        splited = line.split()
        y_test.append(int(splited[0]))
        x_test.append([float(x) for x in splited[i_x:i_y]])
    f.close()

    return x_train, y_train, x_test, y_test

def plot_tsne(data, labels, fig_name):
    print(len(data))
    print(len(labels))
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(data)
    
    x = []
    y = []
    for v in new_values:
        x.append(v[0])
        y.append(v[1])
    colors = {0: 'red', 1: 'blue'}
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i], c=colors[labels[i]])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.title(fig_name)
    plt.savefig("analysis/" + fig_name + ".png", bbox_inches = "tight")
    plt.close() 

def main():
    args = read_args()
    
    x_train, y_train, x_test, y_test = read_features(args.train, args.test, 1, -1)
    plot_tsne(x_train+x_test, y_train+y_test, args.dataset+"_mean_median_sd")
    x_train, y_train, x_test, y_test = read_features(args.train, args.test, 1, 51)
    plot_tsne(x_train+x_test, y_train+y_test, args.dataset+"_mean")
    x_train, y_train, x_test, y_test = read_features(args.train, args.test, 51, 101)
    plot_tsne(x_train+x_test, y_train+y_test, args.dataset+"_median")
    x_train, y_train, x_test, y_test = read_features(args.train, args.test, 101, -1)
    plot_tsne(x_train+x_test, y_train+y_test, args.dataset+"_sd")
    x_train, y_train, x_test, y_test = read_features(args.train, args.test, 1, 101)
    plot_tsne(x_train+x_test, y_train+y_test, args.dataset+"_mean_median")

if __name__ == "__main__":
    main()
