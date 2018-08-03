#!/usr/bin/env python
import os
import yaml
import sys
import re
import numpy as np
import scipy.stats as st

import argparse

from igraph import Graph
from zope.dottedname import resolve
from zipfile import ZipFile

from metrics import (
    SIMPLE_METRICS,
    DegreeMetric,
    MetricCreator,
)

from random_graphs import generate_random_graphs
from graphs import (
    get_cut_graphs,
    get_ranking_result,
    get_ranking_scores,
    get_metric_values,
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


def save_graph_statistics(cutted_graphs, graph, metrics, output_format='.pdf'):
    evader = graph.evader
    scores_table = []
    shifted_scores_table = []
    for metric in metrics:
        output_file = os.path.join(graph.name, metric.name + output_format)
        results = get_ranking_result(cutted_graphs, evader, metric)
        save_metric_ranking_plot(results, metric.name, cutted_graphs.label, output_file)

        scores, shifted_socres = get_ranking_scores(results, metric.name)
        scores_table.append(scores)
        shifted_scores_table.append(shifted_socres)

    output_score_file = os.path.join(
        graph.name, 'scores_table' + output_format
    )
    save_scores_table(scores_table, cutted_graphs.label.upper(), output_score_file)

    output_relative_score_file = os.path.join(
        graph.name, 'relative_scores_table' + output_format
    )
    save_scores_table(shifted_scores_table, output_relative_score_file)


def save_influences(graph_sets, graph):
    ici_values = get_metric_values(
        graph_sets, graph.evader, IndependentCascadeInfluence
    )

    output_ici = os.path.join(graph.name, 'independent_cascade_influence.pdf')
    save_influence_value_plot(
        ici_values, IndependentCascadeInfluence.NAME, graph_sets.label,
        output_file=output_ici
    )

    lti_values = get_metric_values(
        graph_sets, graph.evader, IndependentCascadeInfluence
    )

    output_lti = os.path.join(graph.name, 'linear_threshold_influence.pdf')
    save_influence_value_plot(
        lti_values, LinearThresholdInfluence.NAME, graph_sets.label,
        output_file=output_lti
    )


def generate_specific_graph_report(cutted_graphs, graph, metrics, config):
    try:
        os.mkdir(graph.name)
    except OSError:
        pass

    if metrics:
        save_graph_statistics(cutted_graphs, graph, metrics)

    if config.get('append_influences_plot'):
        save_influences(cutted_graphs, graph)

    with ZipFile(graph.name + '.zip', 'w') as zip_:
        zipdir(graph.name, zip_)


def generate_sampling_report(config, metrics, cut_function, label):
    for random_graphs_cfg in config.get('random_graphs', []):
        algorithm = resolve.resolve(random_graphs_cfg.pop('func'))
        random_graphs_cfg['algorithm'] = algorithm
        ranking_table = get_metrics_statics(
            random_graphs_cfg, metrics, cut_function, label
        )
        ranking_table = {k: calculate_average_and_confidence_interval(v)
                         for k, v in ranking_table.items()}


def get_metrics_statics(random_graphs_cfg, metrics, cut_function, label):
    ranking_table = {
        metric.name: [] for metric in metrics
    }
    for graph in generate_random_graphs(**random_graphs_cfg):
        evader = DegreeMetric(graph).get_max().index
        cutted_graphs = get_cut_graphs(
            graph, evader, 4, cut_function, label=label
        )
        for metric in metrics:
            results = get_ranking_result(cutted_graphs, evader, metric)
            ranking_table[metric.name].append(results)

    return ranking_table


def calculate_average_and_confidence_interval(results):
    new_results = []
    for i in range(len(results[0])):
        new_results.append([])
        for j in range(len(results[0][0])):
            results_from_all_samples = [sample[i][j] for sample in results]
            new_results[i].append(
                (sum(results_from_all_samples) / len(results_from_all_samples),
                 st.t.interval(0.95, len(results_from_all_samples) - 1,
                               loc=np.mean(results_from_all_samples),
                               scale=st.sem(results_from_all_samples)))
            )

    return new_results


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
        generate_specific_graph_report(cutted_graphs, graph, metrics, config)

    generate_sampling_report(config, metrics, cutting_graph_func, label)


if __name__ == "__main__":
    run_program()
