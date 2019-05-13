#! /usr/bin/env python3

from node2vec import Node2Vec

class MyNode2Vec(object):
    def __init__(self, graph):
        self.graph = graph
        self.model  = None

    def initialize_model(self, dim, walk, num_walks):
        self.model = Node2Vec(self.graph, dimensions=dim, walk_length=walk, num_walks=num_walks, workers=1) 

    def train(self):
        self.model.fit(window=10, min_count=1, batch_words=4)

    def print(self):
        #self.model
        pass
