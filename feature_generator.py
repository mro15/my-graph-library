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

def build_graphs(dataset, window, strategy, cut_percentage, emb_dim):
    graphs = []
    weight_cutter = WeightCutter(
        emb_dim=emb_dim,
        dataset=dataset,
        strategy=strategy,
        window_size=window,
        cut_percentage=cut_percentage
    )

    weight_cutter.construct_graphs()
    return weight_cutter

def learn_node2vec_representation(dataset, train_graphs, test_graphs, emb_dim):
    weight = True # TODO: set for false when unweighted graph
    train_emb = []
    test_emb = []

    print("=== STARTING RL IN TRAIN GRAPHS ===")
    for i in range(0, len(train_graphs)):
        rl = RepresentationLearning(
            graph=train_graphs[i],
            method="node2vec",
            weight=weight,
            sentence=dataset.train_data[i],
            emb_dim=emb_dim
        )
        rl.initialize_rl_class()
        rl.representation_method.initialize_model()
        rl.representation_method.train()
        train_emb.append(rl.representation_method.get_embeddings())
    print("=== FINISHED RL IN TRAIN GRAPHS ===")

    print("=== STARTING RL IN TEST GRAPHS ===")
    for i in range(0, len(test_graphs)):
        rl = RepresentationLearning(
            graph=test_graphs[i],
            method="node2vec",
            weight=weight,
            sentence=dataset.test_data[i],
            emb_dim=emb_dim
        )
        rl.initialize_rl_class()
        rl.representation_method.initialize_model()
        rl.representation_method.train()
        test_emb.append(rl.representation_method.get_embeddings())
    print("=== FINISHED RL IN TEST GRAPHS ===")

    return train_emb, test_emb

def write_representation(dataset, train_emb, test_emb, train_file, test_file):
    print("=== WRITING NODE EMBEDDINGS ===")
    with open(train_file + '_x.pkl', 'wb') as outfile:
        pickle.dump(train_emb, outfile)
    with open(train_file + '_y.pkl', 'wb') as outfile:
        pickle.dump(dataset.train_labels, outfile)
    with open(test_file + '_x.pkl', 'wb') as outfile:
        pickle.dump(test_emb, outfile)
    with open(test_file + '_y.pkl', 'wb') as outfile:
        pickle.dump(dataset.test_labels, outfile)

def main():
    args = read_args()
    dataset = read_dataset(args.dataset)
    dataset.pre_process_data()
    
    weight_cutter = build_graphs(
        dataset=dataset,
        window=args.window,
        strategy=args.strategy,
        cut_percentage=args.cut_percent,
        emb_dim=args.emb_dim
    )

    train_graphs = weight_cutter.graph_builder.train_graphs
    test_graphs = weight_cutter.graph_builder.test_graphs

    train_file, test_file = weight_cutter.get_output_files()
    print(train_file, test_file)

    train_emb, test_emb = learn_node2vec_representation(
        dataset=dataset,
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        emb_dim=args.emb_dim
    )

    write_representation(dataset, train_emb, test_emb, train_file, test_file)

if __name__ == "__main__":
    main()
