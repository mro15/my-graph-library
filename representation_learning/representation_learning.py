#! /usr/bin/env python3

from representation_learning.rl_node2vec import MyNode2Vec
import numpy as np

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

    def set_features(self):
        self.mean, self.median, self.standard_deviation = self.representation_method.embeddings_compact()

    def get_features(self):
        mean_str = " ".join(str(e) for e in self.mean)
        median_str = " ".join(str(e) for e in self.median)
        std_str = " ".join(str(e) for e in self.standard_deviation)
        return mean_str + " " + median_str + " "  + std_str + " "

    def print_features(self):
        print("MEAN: ", self.mean)
        print("MEDIAN: ", self.median)
        print("STD: ", self.standard_deviation)


