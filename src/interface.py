#!/usr/bin/env python
import sys
import re

import argparse

from igraph import Graph

from metrics import (
    DegreeMetric,
    BetweennessMetric,
    ClosenessMetric,
    KCoreDecompositionMetric,
    ExtendedKCoreDecompositionMetric,
    NeighborhoodCorenessMetric,
    ExtendedNeighborhoodCorenessMetric,
    EigenVectorMetric,
    SecondOrderDegreeMassMetric,
    AtMost1DegreeAwayShapleyValue,
    AtLeastKNeighborsInCoalitionShapleyValue,
    INGScoreMetric,
    EffectivenessMetric,
    HIndexMetric,
    LeaderRankMetric,
    ClusterRankMetric,
)

from graphs import (
    get_roam_graphs,
    save_metric_ranking_plot,
)


def generate_metric_plots(graph, boss):
    roams = get_roam_graphs(graph, boss, 4, metric=DegreeMetric)
    roams += get_roam_graphs(
        graph, boss, 4, metric=SecondOrderDegreeMassMetric)

    get_metrics_plot(roams, boss, DegreeMetric)
    get_metrics_plot(roams, boss, BetweennessMetric)
    get_metrics_plot(roams, boss, ClosenessMetric)
    get_metrics_plot(roams, boss, EigenVectorMetric)
    get_metrics_plot(roams, boss, SecondOrderDegreeMassMetric)
    get_metrics_plot(roams, boss, KCoreDecompositionMetric)
    get_metrics_plot(roams, boss, ExtendedKCoreDecompositionMetric)
    get_metrics_plot(roams, boss, NeighborhoodCorenessMetric)
    get_metrics_plot(roams, boss, ExtendedNeighborhoodCorenessMetric)
    get_metrics_plot(roams, boss, AtMost1DegreeAwayShapleyValue)
    get_metrics_plot(roams, boss, AtLeastKNeighborsInCoalitionShapleyValue)
    # get_influence_value(roams, boss, IndependentCascadeInfluence)
    # get_influence_value(roams, boss, LinearThresholdInfluence)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        help=(
            "Put here the path to graph."" If this argument is not pass then"
            "the graph is generated randomly."
        ),
    )
    parser.add_argument(
        "--evader",
        help=(
            "Evader is a node in a graph which is applying roam algorithm"
        ),
        type=int,
        required=False,
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        required=True,
    )
    args = parser.parse_args(sys.argv[1:])
    return args


def random_graph(nodes=20):
    return Graph.GRG(nodes, 0.5)


def find_the_boss(graph):
    return graph.degree().index(max(graph.degree()))


regex_edge = re.compile('(\d+) (\d+)')


def load_anet_graph(filename):
    edges = []
    with open(filename) as f_buffer:
        filelines = f_buffer.readlines()
        for line in filelines:
            match = regex_edge.match(line)
            if match:
                groups = match.groups()
                edges.append((int(groups[0]), int(groups[1])))
    return Graph(edges=edges)


def read_graph(filename):
    if filename.endswith('.anet'):
        graph = load_anet_graph(filename)
    else:
        graph = Graph.Read_Edgelist(filename)

    return graph


metric_map = {
    'DegreeMetric': DegreeMetric,
    'BetweennessMetric': BetweennessMetric,
    'ClosenessMetric': ClosenessMetric,
    'HIndexMetric': HIndexMetric,
    'LeaderRankMetric': LeaderRankMetric,
    'ClusterRankMetric': ClusterRankMetric,
    'INGScoreMetric': INGScoreMetric,
    'KCoreDecompositionMetric': KCoreDecompositionMetric,
    'ExtendedKCoreDecompositionMetric': ExtendedKCoreDecompositionMetric,
    'NeighborhoodCorenessMetric': NeighborhoodCorenessMetric,
    'ExtendedNeighborhoodCorenessMetric': ExtendedNeighborhoodCorenessMetric,
    'SecondOrderDegreeMassMetric': SecondOrderDegreeMassMetric,
    'EigenVectorMetric': EigenVectorMetric,
    'EffectivenessMetric': EffectivenessMetric,
    'AtMost1DegreeAwayShapleyValue': AtMost1DegreeAwayShapleyValue,
    'AtLeastKNeighborsInCoalitionShapleyValue':
        AtLeastKNeighborsInCoalitionShapleyValue,
}


def parse_kwargs(input_list):
    kwargs = {}
    kwarg_regex = re.compile('([A-Za-z0-9_]+)=(.+)')
    for item in input_list[1:]:
        match = kwarg_regex.match(item)
        if match:
            groups = match.groups()
            kwargs[groups[0]] = groups[1]
        else:
            sys.stderr.write('Could not parse argument {}'.format(item))
            sys.exit(-1)

    return kwargs


def create_metric(metric_name, graph, boss=None, **kwargs):
    metric_class = metric_map[metric_name]
    boss = boss or 0
    return metric_class(graph, boss, **kwargs)


def run_program():
    # generate_metric_plots(graph, boss)
    args = parse_args()
    graph = read_graph(args.graph)

    evader = args.evader or DegreeMetric(graph).get_max().index
    roams = get_roam_graphs(graph, evader, 4, metric=DegreeMetric)
    metric_class = metric_map[args.metric[0]]
    save_metric_ranking_plot(
        roams, evader, metric_class, **parse_kwargs(args.metric)
    )


if __name__ == "__main__":
    run_program()
