#! /usr/bin/env python3

from os import listdir
import nltk
import string
import collections

class Dataset(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.validation_data = None
        self.validation_labels = None
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.vocabulary = collections.Counter()

    def read_polarity(self):
        dataset = "datasets/polarity/txt_sentoken/"
        data = {}
        data["neg"] = []
        data["pos"] = []
        for i in ["neg", "pos"]:
            files = listdir(dataset+i+"/")
            sentence = ""
            for f in files:
                fp = open(dataset+i+"/"+f, 'r')
                data[i].append(fp.read().split())
        #shuffle
        #change here the division in train, test and validation
        train_neg = data["neg"][:int(len(data["neg"])/2)]
        train_pos = data["pos"][:int(len(data["pos"])/2)]
        self.train_data = train_neg + train_pos
        self.train_labels = [1]*len(train_pos)+[0]*len(train_neg)

        test_neg = data["neg"][int(len(data["neg"])/2):]
        test_pos = data["pos"][int(len(data["pos"])/2):]
        self.test_data = test_neg + test_pos
        self.test_labels = [1]*len(test_pos)+[0]*len(test_neg)

    def pre_process_data(self):
        # decide if filter small words
        clean_texts = []
        for tokens in self.train_data:
            table = str.maketrans('', '', string.punctuation)
            tokens = [w.translate(table) for w in tokens] #remove punctuation
            tokens = [word for word in tokens if word.isalpha()] #remove non alphabetic tokens
            tokens = [w for w in tokens if not w in self.stop_words]
            self.vocabulary.update(tokens)
            clean_texts.append(tokens)
        self.train_data = clean_texts

        clean_texts = []
        for tokens in self.test_data:
            table = str.maketrans('', '', string.punctuation)
            tokens = [w.translate(table) for w in tokens]
            tokens = [word for word in tokens if word.isalpha()]
            tokens = [w for w in tokens if not w in self.stop_words]
            self.vocabulary.update(tokens)
            clean_texts.append(tokens)
        self.test_data = clean_texts

    #remove uncommon words
    def remove_words(self):
        vocab, count = zip(*self.vocabulary.most_common())
        cutoff = count.index(4)
        vocab = set(vocab[:cutoff])
        self.vocabulary = collections.Counter()
        clean_texts = []
        for tokens in self.train_data:
            tokens = [word for word in tokens if word in vocab]         
            self.vocabulary.update(tokens)
            clean_texts.append(tokens)
        self.train_data = clean_texts
        
        clean_texts = []
        for tokens in self.test_data:
            tokens = [word for word in tokens if word in vocab]         
            self.vocabulary.update(tokens)
            clean_texts.append(tokens)
        self.text_data = clean_texts

    def voc_stats(self):
        print("Size: ", len(self.vocabulary))
        print("Most common words")
        print(self.vocabulary.most_common(50))

    def small_debug(self):
        print("=== DEBUG ===")
        print("=== TRAIN === ", len(self.train_data))
        for i in range(0, 3):
            print(self.train_data[i])
            print(self.train_labels[i])
        print("=== TEST === ", len(self.test_data))
        for i in range(0, 3):
            print(self.test_data[i])
            print(self.test_labels[i])
