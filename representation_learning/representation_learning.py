#! /usr/bin/env python3

from representation_learning.rl_node2vec import MyNode2Vec

class RepresentationLearning(object):
    def __init__(self, graph, method):
        self.graph = graph
        self.method = method
        self.representation_method = None
        self.mean = 0
        self.median = 0
        self.standard_deviation = 0

    def initialize_rl_class(self):
        if self.method == 1:
            #print("=== REPRESENTATION LEARNING: NODE2VEC ===")
            self.representation_method = MyNode2Vec(self.graph)
            return True
        else:
            #print("=== REPRESENTATION LEARNING: UNKNOWN ===")
            return False



