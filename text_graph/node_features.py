#! /usr/bin/env python3

import gensim

class NodeFeatures(object):
    def __init__(self, dataset):
        self.model = None
        self.dataset = dataset

    def load_model(self):
        #load model from specific datset 
        pass

    def edge_weight(self, v1, v2):
        cos = self.w2v_model.similarity(v1, v2)
        #print(v1, v2, cos)
        return cos

    def vertex_w2v_features(self, v):
        pass
