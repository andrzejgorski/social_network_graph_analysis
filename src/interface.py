#!/usr/bin/env python
import os
import yaml
import sys
import re

import argparse

from igraph import Graph
from zope.dottedname import resolve
from zipfile import ZipFile

from metrics import (
    SIMPLE_METRICS,
    DegreeMetric,
    MetricCreator,
)

from graphs import (
    get_cut_graphs,
    save_metric_ranking_plot,
    save_scores_table,
    save_influence_value_plot,
)

from influences import (
    IndependentCascadeInfluence,
    LinearThresholdInfluence,
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

        label = config.get('label', '')

        graph.name = graph_name + '_' + label
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


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def save_graph_static(cutted_graphs, graph, metrics, output_format='.pdf'):
    evader = graph.evader
    scores_table = []
    shifted_scores_table = []
    for metric in metrics:
        output_file = os.path.join(graph.name, metric.name + output_format)
        scores, shifted_socres = save_metric_ranking_plot(
            cutted_graphs, evader, metric, output_file
        )
        scores_table.append(scores)
        shifted_scores_table.append(shifted_socres)

    output_score_file = os.path.join(
        graph.name, 'scores_table' + output_format
    )
    save_scores_table(scores_table, output_score_file)

    output_relative_score_file = os.path.join(
        graph.name, 'relative_scores_table' + output_format
    )
    save_scores_table(shifted_scores_table, output_relative_score_file)


def save_influences(cutted_graphs, graph):
    output_ici = os.path.join(graph.name, 'independent_cascade_influence.pdf')
    save_influence_value_plot(
        cutted_graphs, graph.evader, IndependentCascadeInfluence,
        output_file=output_ici
    )

    output_lti = os.path.join(graph.name, 'linear_threshold_influence.pdf')
    save_influence_value_plot(
        cutted_graphs, graph.evader, LinearThresholdInfluence,
        output_file=output_lti
    )


def generate_zipped_raport(cutted_graphs, graph, metrics):
    try:
        os.mkdir(graph.name)
    except OSError:
        pass

    if metrics:
        save_graph_static(cutted_graphs, graph, metrics)

    if config.get('append_influences_plot'):
        save_influences(cutted_graphs, graph)

    with ZipFile(graph.name + '.zip', 'w') as zip_:
        zipdir(graph.name, zip_)


def run_program():
    args = parse_args()
    config = load_config(args.config)
    graphs = load_graphs(config)

    cutting_graph_func = resolve.resolve(
        config.get('cutting_graph_heuristic', 'graphs.remove_one_add_many'))

    label = config.get('label') or 'roam'
    cutted_graph_sets = [
        get_cut_graphs(
            graph, graph.evader, 4,
            function=cutting_graph_func, metric=DegreeMetric, label=label
        )
        for graph in graphs
    ]

    metrics = load_metrics(config)
    for cutted_graphs, graph in zip(cutted_graph_sets, graphs):
        generate_zipped_raport(cutted_graphs, graph, metrics)

if __name__ == "__main__":
    run_program()
