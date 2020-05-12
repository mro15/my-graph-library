#! /usr/bin/env python3

# External imports

# Internal imports
from text_graph.text_graph import TextGraph

class GraphBuilder():

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

    def local_pmi_graph(self):
        """
            Graph with local PMI edge weight
        """
        
        """
            Decidir se:
             - no mesmo for construo todos os grafos, ordeno os pesos e removo as arestas
            Ou se:
             - faço um for pra contruir os grafos, depois outro for que ordena as arestas e
                remove as menores conforme a porcentagem
        """

        pass

    def global_pmi_graph(self):
        """
            Graph with global PMI edge weight
        """

        pass
