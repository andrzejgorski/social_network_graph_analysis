#!/usr/bin/env python
import os
import yaml
import sys
import re

import argparse

from igraph import Graph
from zope.dottedname import resolve

from metrics import (
    SIMPLE_METRICS,
    DegreeMetric,
    MetricCreator,
)

from core import (
    generate_sampling_report,
    generate_specific_graph_raport,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
    )
    args = parser.parse_args(sys.argv[1:])
    return args


def random_graph(nodes=20):
    return Graph.GRG(nodes, 0.5)


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


def get_graph_name(path):
    return path.split(os.sep)[-1][:-5]


def load_graphs(config):
    result = []
    for graph_cfg in config.get('specific_graphs', []):
        graph = read_graph(graph_cfg['path'])
        evader = graph_cfg.get('evader')
        if evader is not None:
            graph.evader = int(evader)
        else:
            graph.evader = DegreeMetric(graph).get_max().index
        graph_name = graph_cfg.get('name')
        if graph_name is None:
            graph_name = get_graph_name(graph_cfg['path'])

        graph.name = graph_name
        result.append(graph)

    return result


def load_config(path):
    with open(path) as stream:
        return yaml.load(stream)


def load_include_metrics(config):
    result = [MetricCreator(metric) for metric in SIMPLE_METRICS]
    for metric_cfg in config.get('include_metrics', []):
        metric_class = resolve.resolve(metric_cfg['name'])
        del metric_cfg['name']
        metric_creator = MetricCreator(metric_class, **metric_cfg)
        result.append(metric_creator)

    return result


def load_exclude_metrics(config):
    result = SIMPLE_METRICS
    for metric_cfg in config.get('exclude_metrics', []):
        metric_class = resolve.resolve(metric_cfg['name'])
        result.remove(metric_class)

    return [MetricCreator(metric) for metric in result]


def load_metrics(config):
    metrics = load_include_metrics(config)

    if not metrics:
        metrics = load_exclude_metrics(config)
    return metrics


def run_program():
    args = parse_args()
    config = load_config(args.config)
    graphs = load_graphs(config)

    metrics = load_metrics(config)
    append_influences = config.get('append_influences_plot')

    for heur_cfg in config.get('cutting_graph_heuristics'):
        cut_graph_func = resolve.resolve(heur_cfg.get('func'))

        label = heur_cfg.get('label')
        for graph in graphs:
            generate_specific_graph_raport(
                graph, metrics, append_influences, cut_graph_func, label
            )

        for random_graphs_cfg in config.get('random_graphs', []):
            generate_sampling_report(
                random_graphs_cfg, metrics, cut_graph_func, label
            )


if __name__ == "__main__":
    run_program()
