#! /usr/bin/env python3

# Internal imports
from weight_cutter.graph_builder import GraphBuilder

class WeightCutter():

    train_grapphs = []
    test_graphs = []

    def __init__(self, **kwargs):
        self.graph_builder = GraphBuilder(**kwargs)

    def construct_graphs(self):
        """
            Build the graphs
        """

        self.graph_builder.print_parameters()

    def make_out_dir(self):
        """
            Creates de output directory if not exists
        """

        pass

    def get_output_files(self):
        """
            Return the train and test output files
        """

        pass
