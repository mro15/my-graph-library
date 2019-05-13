#! /usr/bin/env python3

from node2vec import Node2Vec
import numpy as np

class MyNode2Vec(object):
    def __init__(self, graph):
        self.graph = graph
        self.model  = None
        self.trained_model = None

    def initialize_model(self, dim, walk, num_walks):
        self.model = Node2Vec(self.graph, dimensions=dim, walk_length=walk, num_walks=num_walks, workers=1) 

    def train(self):
        self.trained_model = self.model.fit(window=10, min_count=1, batch_words=4)

    def print(self):
        for i in self.graph.nodes():
            print(i, ": ", self.trained_model.wv[str(i)])

    #this method returns the mean, the median and the standard deviation of the graph node embeddings 
    def embeddings_compact(self):
        emb = []
        for i in self.graph.nodes():
            emb.append(self.trained_model.wv[str(i)])
        return np.mean(emb, axis=0), np.median(emb, axis=0), np.std(emb, axis=0)
