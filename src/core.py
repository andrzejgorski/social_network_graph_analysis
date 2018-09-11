import math
import os
import numpy as np
import scipy.stats as st

from zipfile import ZipFile

from random_graphs import generate_random_graphs

from heuristics import get_cut_graphs
from scores import get_ranking_scores
from charts import (
    save_metric_ranking_plot,
    save_metric_ranking_plot_for_random_graphs,
    save_scores_table,
    save_influence_value_plot,
    save_influence_value_plot_for_random_graphs
)
from influences import (
    IndependentCascadeInfluence,
    LinearThresholdInfluence,
)
from metrics import DegreeMetric


def get_ranking_result(graph_sets, boss, metric_cls):
    return [
        [
            metric_cls(graph, boss).get_node_ranking(boss) for graph in g_set
        ]
        for g_set in graph_sets
    ]


def get_metric_values(graph_sets, boss, metric_cls, **kwargs):
    return [
        [
            metric_cls(graph, boss, **kwargs).apply_metric(boss) for graph in g_set
        ]
        for g_set in graph_sets
    ]


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def save_graph_statistics(cut_graphs, graph, metrics, label, dir_name,
                          output_format='.pdf'):
    evader = graph.evader
    scores_table = []
    shifted_scores_table = []
    for metric in metrics:
        output_file = os.path.join(dir_name, metric.name + output_format)
        results = get_ranking_result(cut_graphs, evader, metric)
        save_metric_ranking_plot(results, metric.name, label, output_file)

        scores, shifted_socres = get_ranking_scores(results, metric.name)
        scores_table.append(scores)
        shifted_scores_table.append(shifted_socres)

    output_score_file = os.path.join(
        dir_name, 'scores_table' + output_format
    )
    save_scores_table(scores_table, label.upper(), output_score_file)

    output_relative_score_file = os.path.join(
        dir_name, 'relative_scores_table' + output_format
    )
    save_scores_table(
        shifted_scores_table, label.upper(), output_relative_score_file)


def save_random_graphs_statistics(results, label, dir_name,
                                  output_format='.pdf'):
    scores_table = []
    shifted_scores_table = []
    for metric, results in results.items():
        output_file = os.path.join(dir_name, metric + output_format)
        save_metric_ranking_plot_for_random_graphs(
            results, metric, label, output_file
        )

        only_average = [[x[0] for x in y] for y in results]

        scores, shifted_socres = get_ranking_scores(only_average, metric)
        scores_table.append(scores)
        shifted_scores_table.append(shifted_socres)

    output_score_file = os.path.join(
        dir_name, 'scores_table' + output_format
    )
    save_scores_table(scores_table, label.upper(), output_score_file)

    output_relative_score_file = os.path.join(
        dir_name, 'relative_scores_table' + output_format
    )
    save_scores_table(
        shifted_scores_table, label.upper(), output_relative_score_file
    )


def save_random_graphs_influences_statistics(results, label, dir_name,
                                             output_format='.pdf'):
    for influence, results in results.items():
        save_influence_value_plot_for_random_graphs(
            results, influence, label, dir_name
        )


def save_influence(influence_function, graph_sets, graph, label, dir_name, influences_sample_size, influences_dummy_alpha):
    values = get_metric_values(graph_sets, graph.evader, influence_function,
                               samplings=influences_sample_size, dummy_alpha=influences_dummy_alpha)
    save_influence_value_plot(values, influence_function.NAME, label, dir_name)


def save_influences(*args):
    save_influence(IndependentCascadeInfluence, *args)
    save_influence(LinearThresholdInfluence, *args)


def generate_specific_graph_raport(graph, metrics, append_influences, influences_sample_size, influences_dummy_alpha,
                                   cut_graph_func, budgets, executions, label):
    dir_name = graph.name + '_' + label
    try:
        os.mkdir(dir_name)
    except OSError:
        pass

    cut_graphs = get_cut_graphs(
        graph, graph.evader, executions, budgets,
        function=cut_graph_func, metric=DegreeMetric,
    )

    if metrics:
        save_graph_statistics(cut_graphs, graph, metrics, label, dir_name)

    if append_influences:
        save_influences(cut_graphs, graph, label, dir_name, influences_sample_size, influences_dummy_alpha)

    with ZipFile(dir_name + '.zip', 'w') as zip_:
        zipdir(dir_name, zip_)


def generate_sampling_report(random_graphs_kwargs, metrics, append_influences,
                             influences_sample_size, influences_dummy_alpha, cut_function,
                             budgets, executions, label):

    dir_name = random_graphs_kwargs['algorithm'].__name__ + '_' + label
    print('Generating reports for ' + dir_name)

    try:
        os.mkdir(dir_name)
    except OSError:
        pass

    ranking_table, influences_table = get_metrics_statistics(
        random_graphs_kwargs, metrics, cut_function, executions, label, budgets,
        append_influences, influences_sample_size, influences_dummy_alpha
    )
    ranking_table = {k: calculate_average_and_confidence_interval(v)
                     for k, v in ranking_table.items()}
    save_random_graphs_statistics(ranking_table, label,
                                  dir_name)

    if append_influences:
        influences_table = {k: calculate_average_and_confidence_interval(v)
                     for k, v in influences_table.items()}
        save_random_graphs_influences_statistics(influences_table, label, dir_name)

    with ZipFile(dir_name + '.zip', 'w') as zip_:
        zipdir(dir_name, zip_)


def get_metrics_statistics(random_graphs_kwargs, metrics, cut_function, executions, label,
                           budgets, append_influences, influences_sample_size, influences_dummy_alpha):
    ranking_table = {
        metric.name: [] for metric in metrics
    }

    influences_table = {
        influence.__name__: [] for influence in
        (LinearThresholdInfluence, IndependentCascadeInfluence)
    }

    for graph in generate_random_graphs(**random_graphs_kwargs):
        evader = DegreeMetric(graph).get_max().index
        cut_graphs = get_cut_graphs(
            graph, evader, executions, budgets, cut_function,
        )

        for metric in metrics:
            results = get_ranking_result(cut_graphs, evader, metric)
            ranking_table[metric.name].append(results)

        if append_influences:
            influence_results = get_influences_results(cut_graphs, evader, influences_sample_size,
                                                       influences_dummy_alpha)
            for inf_results in influence_results:
                influences_table[inf_results[0]].append(inf_results[1])

    return ranking_table, influences_table


def get_influences_results(*args):
    return (
        get_influence_results(LinearThresholdInfluence, *args),
        get_influence_results(IndependentCascadeInfluence, *args)
    )


def get_influence_results(influence_function, cut_graphs, evader, samples, influences_dummy_alpha):
    return (
        influence_function.__name__,
        get_metric_values(cut_graphs, evader,
                          influence_function, samplings=samples, dummy_alpha=influences_dummy_alpha)
    )


def calculate_average_and_confidence_interval(results):
    new_results = []
    for i in range(len(results[0])):
        new_results.append([])
        for j in range(len(results[0][0])):
            results_from_all_samples = [sample[i][j] for sample in results]

            confidence_interval = st.t.interval(
                0.95,
                len(results_from_all_samples) - 1,
                loc=np.mean(results_from_all_samples),
                scale=st.sem(results_from_all_samples)
            )

            if math.isnan(confidence_interval[0]):
                confidence_interval = (
                    results_from_all_samples[0], results_from_all_samples[0]
                )
            new_results[i].append(
                (sum(results_from_all_samples) / len(results_from_all_samples),
                 confidence_interval)
            )

    return new_results
