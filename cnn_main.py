#! /usr/bin/env python3

import argparse
from cnn.my_cnn import My_cnn
import numpy as np
import _pickle as pickle
import sklearn
from scipy.sparse import csr_matrix, lil_matrix
from keras.preprocessing.sequence import pad_sequences

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--dataset', type=str, choices=["imdb", "polarity", "mr", "webkb"], help='dataset name', required=True)   
    parser.add_argument('--method', type=str, choices=["node2vec", "gcn"], help='representation method', required=True)
    parser.add_argument('--strategy', type=str, choices=["no_weight", "pmi_2019", "normalized_pmi", "pmi_1990", "dice", "llr", "chi_square"], help='representation method', required=True)
    parser.add_argument('--window', type=int,  help='window size', required=True)
    parser.add_argument('--emb_dim', type=int,  help='embeddings dimension', required=True)
    parser.add_argument('--pool_type', type=str, choices=["max", "global_max"], help='pooling type', required=True)
    return parser.parse_args()

def padding(train, test, dim):
    m_train = len(max(train, key = lambda i: len(i)))
    m_test = len(max(test, key = lambda i: len(i)))
    m_all = max(m_train, m_test)
    print(m_all)
    pad = np.zeros(dim)
    sparse_all = []
    for i in range(0, len(train)):
        if len(train[i]) < m_all:
            mult = m_all - len(train[i])
            train[i]+= ([pad] * mult)
        sparse_all.append(lil_matrix(train[i]))
    for i in range (0, len(test)):
        if len(test[i]) < m_all:
            mult = m_all - len(test[i])
            test[i]+= ([pad] * mult)
        sparse_all.append(lil_matrix(test[i]))
    return sparse_all

def main():
    args = read_args()

    directory = "graphs/" + args.dataset + "-" + str(args.emb_dim) + "/"
    with open(directory + args.dataset + '_' + args.method + '_' + args.strategy + '_' + str(args.window) + '_' + 'train_x.pkl', 'rb') as infile:
        train_emb = pickle.load(infile)
    with open(directory + args.dataset + '_' + args.method + '_' + args.strategy + '_' + str(args.window) + '_' + 'train_y.pkl', 'rb') as infile:
        train_labels = pickle.load(infile)
    with open(directory + args.dataset + '_' + args.method + '_' + args.strategy + '_' + str(args.window) + '_' + 'test_x.pkl', 'rb') as infile:
        test_emb = pickle.load(infile)
    with open(directory + args.dataset + '_' + args.method + '_' + args.strategy + '_' + str(args.window) + '_' + 'test_y.pkl', 'rb') as infile:
        test_labels = pickle.load(infile)
    

    print(np.array(train_emb).shape)
    print(np.array(test_emb).shape)
    all_x = padding(train_emb, test_emb, args.emb_dim)
    all_y = np.concatenate((np.array(train_labels), np.array(test_labels)), axis=None)

    train_emb = None
    test_emb = None

    print(np.array(all_x).shape)
    print(np.array(all_y).shape)

    mcnn = My_cnn(np.array(all_x), all_y, np.array(all_x[0].A).shape, 2, args.pool_type)
    all_x = None
    all_y = None
    results, results_f1 = mcnn.do_all()

    directory = "results/" + args.dataset + "-" + str(args.emb_dim) + "/"
    with open(directory + args.dataset + '_' + args.method + '_' + args.strategy + '_' + str(args.window) + '.txt', 'w') as f:
        for i in results:
            f.write(str(i) + "\n")
    directory = "results/" + args.dataset + "-" + str(args.emb_dim) + "/"
    with open(directory + "f1_" + args.dataset + '_' + args.method + '_' + args.strategy + '_' + str(args.window) + '.txt', 'w') as f:
        for i in results_f1:
            f.write(str(i) + "\n")

if __name__ == "__main__":
    main()
