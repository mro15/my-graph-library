#! /usr/bin/env python3

# External imports
import os

# Internal imports
from weight_cutter.graph_builder import GraphBuilder


class WeightCutter():

    train_grapphs = []
    test_graphs = []

    def __init__(self, **kwargs):
        self.emb_dim = kwargs.get("emb_dim")
        self.dataset = kwargs.get("dataset")
        self.graph_builder = GraphBuilder(**kwargs)

    def construct_graphs(self):
        """
            Build the graphs
        """

        self.graph_builder.print_parameters()
        self.graph_builder.build_graphs()

    def make_out_dir(self):
        """
            Creates de output directory if not exists
        """

        directory = (
            "graphs/next_level/" + 
            self.dataset.dataset + 
            "-" + 
            str(self.emb_dim) + 
            "/" +
            str(self.graph_builder.cut_percentage) +
            "/"
        )

        if not os.path.exists(directory):
            print("dir not exist, creating ...")
            os.makedirs(directory)

        return directory

    def get_output_files(self):
        """
            Return the train and test output files
        """

        directory = self.make_out_dir()
        train_file = (
            directory +
            self.dataset.dataset + 
            "_node2vec_" + 
            self.graph_builder.strategy +
            "_" +
            str(self.graph_builder.window_size) +
            "_train"
        )
        test_file = (
            directory +
            self.dataset.dataset + 
            "_node2vec_" + 
            self.graph_builder.strategy +
            "_" +
            str(self.graph_builder.window_size) +
            "_test"
        )

        return train_file, test_file
