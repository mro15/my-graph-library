from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import numpy as np

from time_mem.mprof import read_mprofile_file
from text_handler.dataset import Dataset
from weight_cutter.weight_cutter import WeightCutter

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=["polarity", "20ng", "webkb", "r8"],
        help='dataset name',
        required=True
    )   
    parser.add_argument('--window', type=int, help='window size', required=True)
    return parser.parse_args()

def print_results(values, strategy, cut=0):
    memory = np.asarray(values.get('mem_usage')).max()
    t = np.asarray(values.get('timestamp'))
    global_start = float(t[0])
    t = t - global_start
    time = t[-1]
    print(f'METRIC: {strategy}, CUT: {cut}')
    print(f'MAX MEMORY: {memory}')
    print(f'TIME: {time}')
    return time, memory


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


def vertex_and_edges(dataset, window, emb_dim, cut, strategy):

    weight_cutter = build_graphs(
        dataset=dataset,
        window=window,
        strategy=strategy,
        cut_percentage=cut,
        emb_dim=emb_dim
    )

    graphs = (
            weight_cutter.graph_builder.train_graphs +
            weight_cutter.graph_builder.test_graphs
    )
    print(len(graphs))
    edges = [amount.number_of_edges() for amount in graphs]
    nodes = [amount.number_of_nodes() for amount in graphs]
    print(f'==== CUT: {cut} =====')
    print('EDGES')
    print(f'LEN: {len(edges)}, MEAN: {np.mean(edges)}, STD: {np.std(edges)}')
    print('NODES')
    print(f'LEN: {len(nodes)}, MEAN: {np.mean(nodes)}, STD: {np.std(nodes)}')
    weight_cutter = None
    graphs = None
    return np.mean(edges), np.mean(nodes)

def plot_bar_graph(cuts, bar_edges, bar_nodes, bar_time, bar_memory, output_fig):

    #plt.style.use('seaborn')
    plt.style.use('bmh')
    fig = plt.figure()

    print(bar_edges, bar_nodes, bar_time, bar_memory)

    """
    bar_edges = [452.06593692989316, 429.78342455043]
    bar_nodes = [43.11323951003388, 43.11271826948136]
    bar_time = [16695.547300100327, 15434.610500097275]
    bar_memory = [2002.246094, 1565.542969]
    print(bar_edges, bar_nodes, bar_time, bar_memory)
    """

    host = HostAxes(fig, [0.2, 0.1, 0.55, 0.78])
    par1 = ParasiteAxes(host, sharex=host)
    par2 = ParasiteAxes(host, sharex=host)
    par3 = ParasiteAxes(host, sharex=host)
    host.parasites.append(par3)
    host.parasites.append(par1)
    host.parasites.append(par2)

    host.axis["right"].set_visible(False)

    par1.axis["right"].set_visible(True)
    par1.axis["right"].major_ticklabels.set_visible(True)
    par1.axis["right"].label.set_visible(True)

    par2.axis["right2"] = par2.new_fixed_axis(loc="right", offset=(50, 0))
    par3.axis["left"] = par3.new_fixed_axis(loc="left", offset=(-50, 0))

    fig.add_axes(host)

    positions = np.arange(len(cuts))
    width = 0.2

    p1 = host.bar(positions - width, bar_edges, width, label="# edges", color='teal')
    #host.set_xticks(positions)

    p4 = par3.bar(positions, bar_nodes, width, label="# nodes", color='peru')
    p2 = par1.bar(positions + width, bar_memory, width, label="memory", color='steelblue')
    p3 = par2.bar(positions + width*2, bar_time, width, label="time", color='rosybrown')
    

    #host.set_xlim(0, np.asarray(cuts).max())
    host.set_ylim(0, np.asarray(bar_edges).max())
    par3.set_ylim(0, np.asarray(bar_nodes).max())
    par1.set_ylim(0, np.asarray(bar_memory).max())
    par2.set_ylim(0, np.asarray(bar_time).max())

    host.set_xlabel("CUT PERCENTAGE")
    host.set_ylabel("# EDGES")
    par3.set_ylabel("# NODES")
    par1.set_ylabel("MEMORY")
    par2.set_ylabel("TIME")


    host.legend()
    legend_str = [str(cut) for cut in cuts]
    #host.set_xticklabels(legend_str)
    plt.xticks(positions+(width/2), legend_str)
    host.axis["left"].label.set_color('teal')
    par1.axis["right"].label.set_color('steelblue')
    par2.axis["right2"].label.set_color('rosybrown')
    par3.axis["left"].label.set_color('peru')

    #plt.tight_layout()
    plt.savefig(output_fig)
    plt.close()


def main():

    args = read_args()
    dataset = args.dataset
    window = args.window


    print('===== BASELINE RESULTS =====')
    filename = (
        f'time_mem/mprofile.{dataset}.no_weight.{window}.0.dat'
    )
    values = read_mprofile_file(filename=filename)
    base_time, base_memory = print_results(values, 'no_weight')


    dataset_object = read_dataset(args.dataset)
    dataset_object.pre_process_data()
    base_edges, base_nodes = vertex_and_edges(dataset_object, args.window, 100, 0, 'no_weight')
    strategies = ["chi_square", "chi_square_all", "llr", "llr_all", "pmi", "pmi_all"]
    cuts = [0, 5, 10, 20, 30, 50, 70, 80, 90]
    #cuts = [0, 5, 10]

    print('===== PROPOSAL RESULTS =====')
    for strategy in strategies:
        print("\n\n")
        bar_edges = []
        bar_nodes = []
        bar_time = []
        bar_memory = []
        bar_time.append(base_time)
        bar_memory.append(base_memory)
        bar_edges.append(base_edges)
        bar_nodes.append(base_nodes)
        for cut in cuts[1:]:
            filename = (
                f'time_mem/mprofile.{dataset}.{strategy}.{window}.{cut}.dat'
            )
            values = read_mprofile_file(filename=filename)
            time, memory = print_results(values, strategy, cut)
            edges, nodes = vertex_and_edges(dataset_object, args.window, 100, cut, strategy)
            bar_edges.append(edges)
            bar_nodes.append(nodes)
            bar_time.append(time)
            bar_memory.append(memory)
        output_fig = "sac_results/all_" + dataset + "_" + str(args.window) + strategy + ".png"
        plot_bar_graph(cuts, bar_edges, bar_nodes, bar_time, bar_memory, output_fig)

if __name__ == "__main__":
    main()
