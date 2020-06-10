#! /usr/bin/env python3

# External imports
from tqdm import tqdm
import itertools
from nltk.collocations import BigramAssocMeasures as bam
from nltk.collocations import BigramCollocationFinder as bcf

# Internal imports
from text_graph.text_graph import TextGraph

class GraphBuilder():

    train_graphs = []
    test_graphs = []

    #weight local functions
    weight_functions = {
        "pmi": bam.pmi,
        "llr": bam.likelihood_ratio,
        "dice": bam.dice,
        "chi_square": bam.chi_sq
    }

    # weight global functions
    weight_global_functions = {
        "pmi_all": bam.pmi,
        "llr_all": bam.likelihood_ratio,
        "dice_all": bam.dice,
        "chi_square_all": bam.chi_sq
    }

    def __init__(self, **kwargs):
        self.dataset = kwargs.get("dataset")
        self.strategy = kwargs.get("strategy")
        self.window_size = kwargs.get("window_size")
        self.cut_percentage = float(kwargs.get("cut_percentage"))/100.0

    def print_parameters(self):
        """
            Print the parameters used to build the graphs
        """

        print("Dataset:", self.dataset.dataset)
        print("strategy:", self.strategy)
        print("window_size:", self.window_size)
        print("cut_percentage:", self.cut_percentage)

    def get_weight_function(self):
        local_weight = self.weight_functions.get(self.strategy)
        global_weight = self.weight_global_functions.get(self.strategy)

        return local_weight, global_weight

    def build_graphs(self):
        local_weight, global_weight =  self.get_weight_function()
        if local_weight:
            self.local_weighted_graphs(local_weight)
        else:
            self.global_weighted_graphs(global_weight)

        print(len(self.train_graphs), len(self.test_graphs))

    def local_weighted_graphs(self, weight_function):
        """
            Graphs with local edge weight
        """
        
        print("BUILDING GRAPHS FROM TRAIN DATASET")
        progress = tqdm(self.dataset.train_data)
        for i in progress:
            g = TextGraph(self.dataset.dataset)
            if len(i) > self.window_size:
                windows = bcf.from_words(i, window_size=self.window_size)
                edges_weights = windows.score_ngrams(weight_function)
                edges_weights.sort(key=lambda value:value[1])
                cut_point = int(len(edges_weights)*self.cut_percentage)
                edges_weights_cut = edges_weights[cut_point:]
                for pairs in edges_weights_cut:
                    pmi = pairs[1]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if pmi > 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, pmi)
            else:
                if len(i) > 1:
                    windows = bcf.from_words(i, window_size=len(i))
                    try:
                        edges_weights = windows.score_ngrams(weight_function)
                        edges_weights.sort(key=lambda value:value[1])
                        cut_point = int(len(edges_weights)*self.cut_percentage)
                        edges_weights_cut = edges_weights[cut_point:]
                        for pairs in edges_weights_cut:
                            pmi = pairs[1]
                            w1 = pairs[0][0]
                            w2 = pairs[0][1]
                            if pmi > 0:
                                g.add_vertex(w1)
                                g.add_vertex(w2)
                                g.add_weight_edge(w1, w2, pmi)
                    except:
                        i_unique = list(set(i))
                        windows = bcf.from_words(i_unique, window_size=len(i_unique))
                        edges_weights = windows.score_ngrams(weight_function)
                        edges_weights.sort(key=lambda value:value[1])
                        cut_point = int(len(edges_weights)*self.cut_percentage)
                        edges_weights_cut = edges_weights[cut_point:]
                        for pairs in edges_weights_cut:
                            pmi = pairs[1]
                            w1 = pairs[0][0]
                            w2 = pairs[0][1]
                            if pmi > 0:
                                g.add_vertex(w1)
                                g.add_vertex(w2)
                                g.add_weight_edge(w1, w2, pmi)

            if((len(pairs)<1) or (len(g.nodes())==0)):
                g.add_vertex(i[0])
            self.train_graphs.append(g.graph)
        print("FINISHED GRAPHS FROM TRAIN DATASET")
    
        print("BUILDING GRAPHS FROM TEST DATASET")
        progress = tqdm(self.dataset.test_data)
        for i in progress:
            g = TextGraph(self.dataset.dataset)
            if len(i) > self.window_size:
                windows = bcf.from_words(i, window_size=self.window_size)
                edges_weights = windows.score_ngrams(weight_function)
                edges_weights.sort(key=lambda value:value[1])
                cut_point = int(len(edges_weights)*self.cut_percentage)
                edges_weights_cut = edges_weights[cut_point:]
                for pairs in edges_weights_cut:
                    pmi = pairs[1]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if pmi > 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, pmi)
            else:
                if len(i) > 1:
                    windows = bcf.from_words(i, window_size=len(i))
                    try:
                        edges_weights = windows.score_ngrams(weight_function)
                        edges_weights.sort(key=lambda value:value[1])
                        cut_point = int(len(edges_weights)*self.cut_percentage)
                        edges_weights_cut = edges_weights[cut_point:]
                        for pairs in edges_weights_cut:
                            pmi = pairs[1]
                            w1 = pairs[0][0]
                            w2 = pairs[0][1]
                            if pmi > 0:
                                g.add_vertex(w1)
                                g.add_vertex(w2)
                                g.add_weight_edge(w1, w2, pmi)
                    except:
                        i_unique = list(set(i))
                        windows = bcf.from_words(i_unique, window_size=len(i_unique))
                        edges_weights = windows.score_ngrams(weight_function)
                        edges_weights.sort(key=lambda value:value[1])
                        cut_point = int(len(edges_weights)*self.cut_percentage)
                        edges_weights_cut = edges_weights[cut_point:]
                        for pairs in edges_weights_cut:
                            pmi = pairs[1]
                            w1 = pairs[0][0]
                            w2 = pairs[0][1]
                            if pmi > 0:
                                g.add_vertex(w1)
                                g.add_vertex(w2)
                                g.add_weight_edge(w1, w2, pmi)

            if((len(pairs)<1) or (len(g.nodes())==0)):
                g.add_vertex(i[0])
            self.test_graphs.append(g.graph)
        print("FINISHED GRAPHS FROM TEST DATASET")

    def global_weighted_graphs(self, weight_function):
        """
            Graphs with global edge weight
        """
        windows = bcf.from_words(self.all_docs_to_one_tokens_list(), window_size=self.window_size)
        weight_all = dict(windows.score_ngrams(weight_function))
        print("BUILDING GRAPHS FROM TRAIN DATASET")
        progress = tqdm(self.dataset.train_data)
        for i in progress:
            g = TextGraph(self.dataset.dataset)
            if len(i) > self.window_size:
                local_windows = bcf.from_words(i, window_size=self.window_size)
                edges_weights = []
                for pairs in local_windows.score_ngrams(weight_function):
                    edges_weights.append((pairs[0], weight_all[pairs[0]]))
                edges_weights.sort(key=lambda value:value[1])
                cut_point = int(len(edges_weights)*self.cut_percentage)
                edges_weights_cut = edges_weights[cut_point:]
                for pairs in edges_weights_cut:
                    pmi = pairs[1]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if pmi > 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, pmi)
            else:
                if len(i) > 1:
                    local_windows = bcf.from_words(i, window_size=len(i))
                    edges_weights = []
                    try:
                        for pairs in local_windows.score_ngrams(weight_function):
                            edges_weights.append((pairs[0], weight_all[pairs[0]]))
                        edges_weights.sort(key=lambda value:value[1])
                        cut_point = int(len(edges_weights)*self.cut_percentage)
                        edges_weights_cut = edges_weights[cut_point:]
                        for pairs in edges_weights_cut:
                            pmi = pairs[1]
                            w1 = pairs[0][0]
                            w2 = pairs[0][1]
                            if pmi > 0:
                                g.add_vertex(w1)
                                g.add_vertex(w2)
                                g.add_weight_edge(w1, w2, pmi)
                    except:
                        edges_weights = []
                        i_unique = list(set(i)) # calculate only for unique tokens
                        local_windows = bcf.from_words(i_unique, window_size=len(i_unique))
                        for pairs in local_windows.score_ngrams(weight_function):
                            edges_weights.append((pairs[0], weight_all[pairs[0]]))
                        edges_weights.sort(key=lambda value:value[1])
                        cut_point = int(len(edges_weights)*self.cut_percentage)
                        for pairs in edges_weights_cut:
                            pmi = pairs[1]
                            w1 = pairs[0][0]
                            w2 = pairs[0][1]
                            if pmi > 0:
                                g.add_vertex(w1)
                                g.add_vertex(w2)
                                g.add_weight_edge(w1, w2, pmi)

            if((len(pairs)<1) or (len(g.nodes())==0)):
                g.add_vertex(i[0])
            self.train_graphs.append(g.graph)
        print("FINISHED GRAPHS FROM TRAIN DATASET")

        print("BUILDING GRAPHS FROM TEST DATASET")
        progress = tqdm(self.dataset.test_data)
        for i in progress:
            g = TextGraph(self.dataset.dataset)
            if len(i) > self.window_size:
                local_windows = bcf.from_words(i, window_size=self.window_size)
                edges_weights = []
                for pairs in local_windows.score_ngrams(weight_function):
                    edges_weights.append((pairs[0], weight_all[pairs[0]]))
                edges_weights.sort(key=lambda value:value[1])
                cut_point = int(len(edges_weights)*self.cut_percentage)
                edges_weights_cut = edges_weights[cut_point:]
                for pairs in edges_weights_cut:
                    pmi = pairs[1]
                    w1 = pairs[0][0]
                    w2 = pairs[0][1]
                    if pmi > 0:
                        g.add_vertex(w1)
                        g.add_vertex(w2)
                        g.add_weight_edge(w1, w2, pmi)
            else:
                if len(i) > 1:
                    local_windows = bcf.from_words(i, window_size=len(i))
                    edges_weights = []
                    try:
                        for pairs in local_windows.score_ngrams(bam.pmi):
                            edges_weights.append((pairs[0], weight_all[pairs[0]]))
                        edges_weights.sort(key=lambda value:value[1])
                        cut_point = int(len(edges_weights)*self.cut_percentage)
                        edges_weights_cut = edges_weights[cut_point:]
                        for pairs in edges_weights_cut:
                            pmi = pairs[1]
                            w1 = pairs[0][0]
                            w2 = pairs[0][1]
                            if pmi > 0:
                                g.add_vertex(w1)
                                g.add_vertex(w2)
                                g.add_weight_edge(w1, w2, pmi)
                    except:
                        edges_weights = []
                        i_unique = list(set(i)) # calculate only for unique tokens
                        local_windows = bcf.from_words(i_unique, window_size=len(i_unique))
                        for pairs in local_windows.score_ngrams(weight_function):
                            edges_weights.append((pairs[0], weight_all[pairs[0]]))
                        edges_weights.sort(key=lambda value:value[1])
                        cut_point = int(len(edges_weights)*self.cut_percentage)
                        for pairs in edges_weights_cut:
                            pmi = pairs[1]
                            w1 = pairs[0][0]
                            w2 = pairs[0][1]
                            if pmi > 0:
                                g.add_vertex(w1)
                                g.add_vertex(w2)
                                g.add_weight_edge(w1, w2, pmi)

            if((len(pairs)<1) or (len(g.nodes())==0)):
                g.add_vertex(i[0])
            self.test_graphs.append(g.graph)
        print("FINISHED GRAPHS FROM TEST DATASET")
                
    def all_docs_to_one_tokens_list(self):
        """
            Converts all documents from dataset to a single document
        """
        docs = self.dataset.train_data+self.dataset.test_data
        token_list = list(itertools.chain(*docs))
        return token_list

