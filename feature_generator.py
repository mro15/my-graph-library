#! /usr/bin/env python3

# External imports
import argparse
import _pickle as pickle

# Internal imports
from text_handler.dataset import Dataset
from weight_cutter.weight_cutter import WeightCutter
from representation_learning.representation_learning import RepresentationLearning

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=["polarity", "webkb", "20ng", "r8"],
        help='dataset name', required=True
    ) 
    parser.add_argument(
        '--strategy',
        type=str,
        choices=[
            "no_weight",
            "freq",
            "freq_all",
            "pmi",
            "pmi_all",
            "dice",
            "dice_all",
            "llr",
            "llr_all",
            "chi_square",
            "chi_square_all"
        ],
        help='representation method',
        required=True
    )
    parser.add_argument('--window', type=int,  help='window size', required=True)
    parser.add_argument(
        '--cut_percent',
        type=int,
        help='percentage of edges to cut',
        required=True
    )
    parser.add_argument('--emb_dim', type=int,  help='embeddings dimension', required=True)
    return parser.parse_args()

def read_dataset(dataset):
    d = Dataset(dataset)
    dataset_readers={
        "polarity": "read_polarity",
        "webkb": "read_webkb",
        "r8": "read_r8",
        "20ng": "read_20ng"
    }
    read_function = getattr(d, dataset_readers.get(dataset))
    read_function()
    return d

def build_graphs(dataset, window, strategy, cut_percentage):
    graphs = []
    weight_cut = WeightCutter(
        dataset=dataset,
        strategy=strategy,
        window_size=window,
        cut_percentage=cut_percentage
    )

    weight_cut.construct_graphs()
    return graphs

def main():
    args = read_args()
    dataset = read_dataset(args.dataset)
    
    graphs = build_graphs(dataset, args.window, args.strategy, args.cut_percent)

if __name__ == "__main__":
    main()
