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

    #weight functions
    weight_functions = {
        "pmi": bam.pmi,
        "llr": bam.likelihood_ratio,
        "dice": bam.dice,
        "chi_square": bam.chi_sq
    }

    def __init__(self, **kwargs):
        self.dataset = kwargs.get("dataset")
        self.strategy = kwargs.get("strategy")
        self.window_size = kwargs.get("window_size")
        self.cut_percentage = kwargs.get("cut_porcentage")

    def print_parameters(self):
        """
            Print the parameters used to build the graphs
        """

        print("Dataset:", self.dataset.dataset)
        print("strategy:", self.strategy)
        print("window_size:", self.window_size)
        print("cut_percentage:", self.cut_percentage)

    def get_weight_function(self):
        return self.weight_functions.get(self.strategy)

    def build_graphs(self):
        # todo: separate between local and global strategies
        self.local_weighted_graphs(self.get_weight_function())

        print(len(self.train_graphs), len(self.test_graphs))

    def local_weighted_graphs(self, weight_function):
        """
            Graphs with local PMI edge weight
        """
        
        """
            Decidir se:
             - no mesmo for construo todos os grafos, ordeno os pesos e removo as arestas
            Ou se:
             - faço um for pra contruir os grafos, depois outro for que ordena as arestas e
                remove as menores conforme a porcentagem
        """

        print("BUILDING GRAPHS FROM TRAIN DATASET")
        progress = tqdm(self.dataset.train_data)
        for i in progress:
            g = TextGraph(self.dataset.dataset)
            if len(i) > self.window_size:
                windows = bcf.from_words(i, window_size=self.window_size)
                for pairs in windows.score_ngrams(weight_function):
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
                    for pairs in windows.score_ngrams(weight_function):
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
                for pairs in windows.score_ngrams(weight_function):
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
                    for pairs in windows.score_ngrams(weight_function):
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

    def global_pmi_graphs(self):
        """
            Graphs with global PMI edge weight
        """

        pass
