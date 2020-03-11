#! /usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse
from text_graph.text_graph import TextGraph
from text_handler.dataset import Dataset
import numpy as np
import networkx as nx
import threshold_handler.analysis as tan

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--strategy', action="append", help='methods to compare', required=True)
    return parser.parse_args()

def main():
    args = read_args()
    datasets = ["polarity", "webkb", "20ng"]
    for dataset in datasets:
        d = Dataset(dataset)
        d.read_20ng()
        d.pre_process_data()
        for window in [4, 7, 12, 20]:
            for strategy in args.strategy:
                output_name = "threshold_handler/graphics/" + dataset + "_" + strategy + "_" + str(window) + "_local.png"
                tan.histogram_strategy_local(d, window, strategy, output_name)
                output_name = "threshold_handler/graphics/" + dataset + "_" + strategy + "_" + str(window) + "_global.png"
                tan.histogram_strategy_global(d, window, strategy, output_name)

    
if __name__ == "__main__":
    main()