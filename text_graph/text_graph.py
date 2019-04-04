#! /usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt

class TextGraph(object):
    def __init__(self, name):
        self.graph = nx.Graph(name=name)

    def get_name(self):
        return self.graph.graph["name"]

    def add_vertex(self, v):
        self.graph.add_node(v)

    def add_edge(self, v1, v2, w):
        self.graph.add_edge(v1, v2, weight=w)

    def plot_graph(self):
        options = {'node_color': 'lightskyblue', 'node_size': 10000, 'with_labels': 'True'}
        edge_labels = nx.get_edge_attributes(self.graph,'weight')
        pos=nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, **options)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.show()
