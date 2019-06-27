#! /usr/bin/env python3

import argparse
from cnn.my_cnn import My_cnn
import numpy as np
import _pickle as pickle
import sklearn

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--dataset', type=str, choices=["imdb", "polarity"], help='dataset name', required=True)   
    parser.add_argument('--method', type=str, choices=["node2vec", "gcn"], help='representation method', required=True)
    parser.add_argument('--strategy', type=str, choices=["no_weight", "pmi"], help='representation method', required=True)
    parser.add_argument('--window', type=int,  help='window size', required=True)
    return parser.parse_args()

def padding(train, test):
    m_train = len(max(train, key = lambda i: len(i)))
    m_test = len(max(test, key = lambda i: len(i)))
    m_all = max(m_train, m_test)
    pad = np.zeros(50)
    for i in range(0, len(train)):
        if len(train[i]) < m_all:
            mult = m_all - len(train[i])
            train[i]+= ([pad] * mult)
    for i in range (0, len(test)):
        if len(test[i]) < m_all:
            mult = m_all - len(test[i])
            test[i]+= ([pad] * mult)
    print(m_all)
    print(train.shape)
    return train, test

def main():
    args = read_args()

    with open('graphs/' + args.dataset + '_' + args.method + '_' + args.strategy + '_' + 'train_x.pkl', 'rb') as infile:
        train_emb = pickle.load(infile)
    with open('graphs/' + args.dataset + '_' + args.method + '_' + args.strategy + '_' + 'train_y.pkl', 'rb') as infile:
        train_labels = pickle.load(infile)
    with open('graphs/' + args.dataset + '_' + args.method + '_' + args.strategy + '_' + 'test_x.pkl', 'rb') as infile:
        test_emb = pickle.load(infile)
    with open('graphs/' + args.dataset + '_' + args.method + '_' + args.strategy + '_' + 'test_y.pkl', 'rb') as infile:
        test_labels = pickle.load(infile)
    

    print(np.array(train_emb).shape)
    print(np.array(test_emb).shape)
    train_emb, test_emb = padding(train_emb, test_emb)
    print(np.array(train_emb).shape)
    print(np.array(test_emb).shape)

    train_emb, train_labels = sklearn.utils.shuffle(train_emb, train_labels, random_state=0)
    test_emb, test_labels = sklearn.utils.shuffle(test_emb, test_labels, random_state=0)

    print(np.array(train_emb).shape)
    print(np.array(test_emb).shape)

    mcnn = My_cnn(train_emb, train_labels, test_emb, test_labels, (len(train_emb[0]),50), 2)
    mcnn.do_all()

if __name__ == "__main__":
    main()
