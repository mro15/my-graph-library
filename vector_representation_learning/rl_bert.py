#! /usr/bin/env python3

from bert_embedding import BertEmbedding
import numpy as np

class MyBert(object):
    def __init__(self):
        self.bert = BertEmbedding()

    def get_embeddings(self, sentence):
        s = [self.tokens_to_sentence(sentence)]
        emb = self.bert(s)
        self.sentence_compact(emb)

    def tokens_to_sentence(self, tokens):
        sentence = ""
        for t in tokens:
            sentence = sentence + t + " "
        return sentence

    def sentence_compact(self, embedding):
        #return np.mean(emb, axis=0), np.median(emb, axis=0), np.std(emb, axis=0)
        pass

