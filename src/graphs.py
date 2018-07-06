from igraph import Graph
from functools import partial
from random import choice

import matplotlib.pyplot as plt
from igraph import Graph
from influences import (
    IndependentCascadeInfluence,
    LinearThresholdInfluence,
)

from metrics import (
    DegreeMetric,
    BetweennessMetric,
    ClosenessMetric,
    KCoreDecompositionMetric,
    ExtendedKCoreDecompositionMetric,
    NeighborhoodCorenessMetric,
    ExtendedNeighborhoodCorenessMetric,
    EffectivenessMetric,
    EigenVectorMetric,
    SecondOrderDegreeMassMetric,
    AtMost1DegreeAwayShapleyValue,
    AtLeastKNeighborsInCoalitionShapleyValue,
    INGScoreMetric,
)
from scores import (
    calculate_integral_score
)


def find_the_boss(graph):
    return graph.degree().index(max(graph.degree()))


def load_graph(filename):
    return Graph.Read_Lgl(filename)


def remove_one_add_many(graph, evader, b, metric):
    graph = graph.copy()

    # step 1
    evader_neighbors = graph.vs[evader].neighbors()
    if len(evader_neighbors) == 0:
        raise StopIteration()

    graph_metric = metric(graph)
    del_neigh = graph_metric.get_max(evader_neighbors)
    graph.delete_edges([(evader, del_neigh.index)])

    # step 2
    try:
        brokers = graph_metric.get_nmin(b - 1, list(set(evader_neighbors) - set(del_neigh.neighbors())))
    except Exception:
        return graph
    graph.add_edges([(del_neigh.index, broker.index) for broker in brokers])

    return graph


def get_roam_graphs(graph, boss, excecutions, metric=DegreeMetric):

    def apply_with_b(graph, evader, b, executions):
        graphs = [graph]
        for _ in range(executions):
            try:
                graph = remove_one_add_many(graph, evader, b, metric)
            except StopIteration:
                break
            graphs.append(graph)
        return graphs

    roam1 = apply_with_b(graph, boss, 1, excecutions)
    roam2 = apply_with_b(graph, boss, 2, excecutions)
    roam3 = apply_with_b(graph, boss, 3, excecutions)
    roam4 = apply_with_b(graph, boss, 4, excecutions)
    return roam1, roam2, roam3, roam4


def save_metric_ranking_plot(roams, boss, metric_cls, output_format='.jpeg',
                             **kwargs):

    def get_metrics(node, graphs, metric):
        return [metric_cls(graph, boss, **kwargs).get_node_ranking(node)
                for graph in graphs]

    plt.figure()
    graph = roams[0][0]
    metric = metric_cls(graph, boss, **kwargs)
    plt.title(metric.NAME)

    results = [get_metrics(boss, roam, metric_cls) for roam in roams]

    for i in range(len(results)):
        label = 'roam' + str(i + 1)
        print("Integral score: {}, {}: {}".format(metric.NAME, label, calculate_integral_score(results[i])))
        plt.plot(results[i], label=label)

    plt.legend(loc=2)
    plt.xlabel("iterations")
    plt.ylabel("ranking")
    plt.savefig(metric.NAME + output_format)


def get_influence_value(roams, boss, influence, output_format='.jpeg'):

    def get_metrics(node, graphs, influence):
        return [influence(graph, samplings=30000).apply_metric(node) for graph in graphs]

    plt.figure()
    plt.title(influence.NAME)

    plt.plot(get_metrics(boss, roams[0], influence), label='roam1')
    # plt.plot(get_metrics(boss, roams[4], influence), label='roam1eig')
    plt.plot(get_metrics(boss, roams[1], influence), label='roam2')
    # plt.plot(get_metrics(boss, roams[5], influence), label='roam2eig')
    plt.plot(get_metrics(boss, roams[2], influence), label='roam3')
    # plt.plot(get_metrics(boss, roams[6], influence), label='roam3eig')
    plt.plot(get_metrics(boss, roams[3], influence), label='roam4')
    # plt.plot(get_metrics(boss, roams[7], influence), label='roam4eig')

    plt.legend(loc=3)
    plt.xlabel("iterations")
    plt.ylabel("value")
    plt.savefig(influence.NAME + output_format)


def generate_metric_plots(graph, boss):
    roams = get_roam_graphs(graph, boss, 4, metric=DegreeMetric)
    # roams += get_roam_graphs(graph, boss, 4, metric=SecondOrderDegreeMassMetric)

    save_metric_ranking_plot(roams, boss, DegreeMetric)
    save_metric_ranking_plot(roams, boss, BetweennessMetric)
    save_metric_ranking_plot(roams, boss, ClosenessMetric)
    save_metric_ranking_plot(roams, boss, EigenVectorMetric)
    save_metric_ranking_plot(roams, boss, SecondOrderDegreeMassMetric)
    save_metric_ranking_plot(roams, boss, KCoreDecompositionMetric)
    save_metric_ranking_plot(roams, boss, ExtendedKCoreDecompositionMetric)
    save_metric_ranking_plot(roams, boss, NeighborhoodCorenessMetric)
    save_metric_ranking_plot(roams, boss, ExtendedNeighborhoodCorenessMetric)
    save_metric_ranking_plot(roams, boss, AtMost1DegreeAwayShapleyValue)
    save_metric_ranking_plot(roams, boss, AtLeastKNeighborsInCoalitionShapleyValue)
    # save_metric_ranking_plot(roams, boss, IndependentCascadeInfluence)
    # save_metric_ranking_plot(roams, boss, LinearThresholdInfluence)
    # save_metric_ranking_plot(roams, boss, EffectivenessMetric)
    # save_metric_ranking_plot(roams, boss, EffectivenessMetric, step_numbers=2)
    # save_metric_ranking_plot(roams, boss, EffectivenessMetric, step_numbers=3)
    save_metric_ranking_plot(roams, boss, INGScoreMetric, benchmark_centrality=KCoreDecompositionMetric, iterations=1, linear_transformation=INGScoreMetric.get_adjacency)
    save_metric_ranking_plot(roams, boss, INGScoreMetric, benchmark_centrality=NeighborhoodCorenessMetric, iterations=1, linear_transformation=INGScoreMetric.get_adjacency)
    save_metric_ranking_plot(roams, boss, INGScoreMetric, benchmark_centrality=DegreeMetric, iterations=1, linear_transformation=INGScoreMetric.get_adjacency)
