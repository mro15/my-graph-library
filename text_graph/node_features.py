#! /usr/bin/env python3

import gensim

class NodeFeatures(object):
    def __init__(self):
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    def edge_weight(self, v1, v2):
        cos = self.w2v_model.similarity(v1, v2)
        #print(v1, v2, cos)
        return cos

    def vertex_w2v_features(self, v):
        pass
