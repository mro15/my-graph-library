#! /usr/bin/env python3

from node2vec import Node2Vec
import numpy as np

class MyNode2Vec(object):
    def __init__(self, graph, weight, sentence, dim, walk_length):
        self.graph = graph
        self.model  = None
        self.trained_model = None
        self.dim = dim
        self.walk_length = walk_length
        self.num_walks = walk_length
        self.workers = 4
        self.weight = weight
        self.sentence = sentence
        self.p = 1
        self.q = 0.25


    def initialize_model(self):
        if self.weight:
            self.model = Node2Vec(self.graph, dimensions=self.dim, p=self.p, q=self.q, walk_length=self.walk_length, num_walks=self.num_walks, weight_key='weight', workers=self.workers, quiet=True) 
        else:
            self.model = Node2Vec(self.graph, dimensions=self.dim, p=self.p, q=self.q, walk_length=self.walk_length, num_walks=self.num_walks, workers=self.workers, quiet=True) 

    def train(self):
        self.trained_model = self.model.fit(window=10, min_count=1, batch_words=4)

    def get_embeddings(self):
        emb = []
        for i in self.sentence:
            #check if word is in vocabulary
            if i in self.trained_model.wv:
                emb.append(self.trained_model.wv[str(i)])
        return emb

    def debug(self):
        print(self.sentence)
        for i in self.sentence:
            print(i, ": ", self.trained_model.wv[str(i)])
        for i in self.graph.nodes():
            print(i, ": ", self.trained_model.wv[str(i)])

    #this method returns the mean, the median and the standard deviation of the graph node embeddings 
    def embeddings_compact(self):
        emb = []
        for i in self.sentence:
            emb.append(self.trained_model.wv[str(i)])
        return np.mean(emb, axis=0), np.median(emb, axis=0), np.std(emb, axis=0)
