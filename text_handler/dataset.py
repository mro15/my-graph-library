#! /usr/bin/env python3

from os import listdir
import nltk
import string
import collections
import re

class Dataset(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.vocabulary = collections.Counter()
        self.classes = 0

    def read_ohsumed(self):
        dataset = "datasets/ohsumed/"
        data_sentence = []
        data_label = []
        for i in ["train", "test"]:
            classes = [f for f in listdir(dataset+i)]
            for c in classes:
                files = listdir(dataset+i+"/"+c)
                for f in files:
                    fp = open(dataset+i+"/"+c+"/"+f, encoding='latin1', mode='r')
                    data_sentence.append(fp.read().split())
                    data_label.append(c)
        labels_map = {}
        for i, label in enumerate(sorted(list(set(data_label)))):
            labels_map.update({label:i})
        data_label_int = []
        for i in range(0, len(data_label)):
            data_label_int.append(labels_map[data_label[i]])
        half = int(len(data_sentence)/2)
        self.train_data = data_sentence[:half]
        self.train_labels = data_label_int[:half]
        self.test_data = data_sentence[half:]
        self.test_labels = data_label_int[half:]
        self.classes = len(list(labels_map))
        print(len(self.train_data), len(self.train_labels))
        print(len(self.test_data), len(self.test_labels))
        print(labels_map, self.classes)

    def read_r8(self):
        dataset = "datasets/r8/"
        data_sentence = []
        data_label = []
        for i in ["train", "test"]:
            fp = open(dataset+i+".txt", 'r')
            for lines in fp.readlines():
                line = lines.split()
                if len(line[1:]) > 0:
                    data_label.append(line[0])
                    data_sentence.append(line[1:])
        labels_map = {}
        for i, label in enumerate(sorted(list(set(data_label)))):
            labels_map.update({label:i})
        data_label_int = []
        for i in range(0, len(data_label)):
            data_label_int.append(labels_map[data_label[i]])
        half = int(len(data_sentence)/2)
        self.train_data = data_sentence[:half]
        self.train_labels = data_label_int[:half]
        self.test_data = data_sentence[half:]
        self.test_labels = data_label_int[half:]
        self.classes = 8

    def read_20ng(self):
        dataset = "datasets/20ng/"
        data_sentence = []
        data_label = []
        for i in ["train", "test"]:
            classes = [f for f in listdir(dataset+i)]
            for c in classes:
                files = listdir(dataset+i+"/"+c)
                for f in files:
                    fp = open(dataset+i+"/"+c+"/"+f, encoding='latin1', mode='r')
                    data_sentence.append(fp.read().split())
                    data_label.append(c)
        labels_map = {}
        for i, label in enumerate(sorted(list(set(data_label)))):
            labels_map.update({label:i})
        data_label_int = []
        for i in range(0, len(data_label)):
            data_label_int.append(labels_map[data_label[i]])
        half = int(len(data_sentence)/2)
        self.train_data = data_sentence[:half]
        self.train_labels = data_label_int[:half]
        self.test_data = data_sentence[half:]
        self.test_labels = data_label_int[half:]
        self.classes = len(list(labels_map))
        #print(len(self.train_data), len(self.train_labels))
        #print(len(self.test_data), len(self.test_labels))
        #print(labels_map, self.classes)

    def read_webkb(self):
        dataset = "datasets/webkb/"
        data_sentence = []
        data_label = []
        for i in ["train", "test"]:
            fp = open(dataset+i+".txt", 'r')
            for lines in fp.readlines():
                line = lines.split()
                if len(line[1:]) > 0:
                    data_label.append(line[0])
                    data_sentence.append(line[1:])
        labels_map = {}
        for i, label in enumerate(sorted(list(set(data_label)))):
            labels_map.update({label:i})
        data_label_int = []
        for i in range(0, len(data_label)):
            data_label_int.append(labels_map[data_label[i]])
        half = int(len(data_sentence)/2)
        self.train_data = data_sentence[:half]
        self.train_labels = data_label_int[:half]
        self.test_data = data_sentence[half:]
        self.test_labels = data_label_int[half:]
        self.classes = 4

    def read_polarity(self):
        dataset = "datasets/polarity/txt_sentoken/"
        data_sentence = []
        data_label = []
        for i in ["neg", "pos"]:
            files = listdir(dataset+i+"/")
            for f in files:
                fp = open(dataset+i+"/"+f, 'r')
                data_label.append(i)
                data_sentence.append(fp.read().split())
        labels_map = {}
        for i, label in enumerate(sorted(list(set(data_label)))):
            labels_map.update({label:i})
        data_label_int = []
        for i in range(0, len(data_label)):
            data_label_int.append(labels_map[data_label[i]])
        half = int(len(data_sentence)/2)
        self.train_data = data_sentence[:half]
        self.train_labels = data_label_int[:half]
        self.test_data = data_sentence[half:]
        self.test_labels = data_label_int[half:]
        self.classes = 2

    def read_imdb(self):
        dataset = "datasets/imdb/aclImdb/"
        data = {}
        data["train"] = {}
        data["train"]["neg"] = []
        data["train"]["pos"] = []
        data["test"] = {}
        data["test"]["neg"] = []
        data["test"]["pos"] = []
        for i in ["train", "test"]:
            for j in ["pos", "neg"]:
                files = listdir(dataset+i+"/"+j+"/")
                for f in files:
                    fp = open(dataset+i+"/"+j+"/"+f, 'r')
                    data[i][j].append(fp.read().split())
        
        self.train_data = data["train"]["neg"] + data["train"]["pos"]
        self.train_labels = [0]*len(data["train"]["neg"]) + [1]*len(data["train"]["pos"])
        self.test_data = data["test"]["neg"] + data["test"]["pos"]
        self.test_labels = [0]*len(data["test"]["neg"]) + [1]*len(data["test"]["pos"])
        self.classes = 2

    def read_mr(self):
        dataset = "datasets/rt-polaritydata/rt-polarity."
        data = {}
        data["neg"] = []
        data["pos"] = []
        for i in ["neg", "pos"]:
            fp = open(dataset+i)
            lines = fp.readlines()
            for l in lines:
                data[i].append(l.split())

        len_train_neg = int(len(data["neg"]) * 0.5)
        len_train_pos = int(len(data["pos"]) * 0.5)
        train_neg = data["neg"][:len_train_neg]
        train_pos = data["pos"][:len_train_pos]
        self.train_data = train_pos + train_neg
        self.train_labels = [1]*len(train_pos)+[0]*len(train_neg)

        test_neg = data["neg"][len_train_neg:]
        test_pos = data["pos"][len_train_pos:]
        self.test_data = test_pos + test_neg
        self.test_labels = [1]*len(test_pos)+[0]*len(test_neg)
        self.classes = 2

    def pre_process_data(self):
        print("PRE PROCESS: START")
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        stem = nltk.stem.PorterStemmer()
        lem = nltk.stem.WordNetLemmatizer()
        clean_texts = []
        for token in self.train_data:
            tokens = ""
            for w in token:
                tokens =  tokens + w + " "
            tokens = tokens.lower()
            #tokens = self.clean_str(tokens)
            tokens = tokenizer.tokenize(tokens)
            tokens = [word for word in tokens if word.isalpha() and (word!="br")] #remove non alphabetic tokens
            tokens = [w for w in tokens if not w in self.stop_words]
            tokens = [stem.stem(w) for w in tokens]
            #tokens = [lem.lemmatize(w) for w in tokens]
            self.vocabulary.update(tokens)
            clean_texts.append(tokens)
        self.train_data = clean_texts

        clean_texts = []
        for token in self.test_data:
            tokens = ""
            for w in token:
                tokens = tokens + w + " "
            #tokens = self.clean_str(tokens)
            tokens = tokens.lower()
            tokens = tokenizer.tokenize(tokens)
            tokens = [word for word in tokens if word.isalpha() and (word!="br")]
            tokens = [w for w in tokens if not w in self.stop_words]
            tokens = [stem.stem(w) for w in tokens]
            #tokens = [lem.lemmatize(w) for w in tokens]
            self.vocabulary.update(tokens)
            clean_texts.append(tokens)
        self.test_data = clean_texts
        print("PRE PROCESS: END")

    def clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

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
        print("====== Vocabulary ======")
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
