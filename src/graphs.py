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
    EigenVectorMetric,
    SecondOrderDegreeMassMetric,
)


def random_graph(nodes=20):
    return Graph.GRG(nodes, 0.5)


def find_the_boss(graph):
    return graph.evcent().index(1)


def remove_one_add_many(graph, evader, b, metric=DegreeMetric):
    graph = graph.copy()

    # step 1
    evader_neighbors = graph.vs[evader].neighbors()
    if len(evader_neighbors) == 0:
        raise StopIteration()

    graph_metric = metric(graph)
    del_neigh = graph_metric.get_max(evader_neighbors)
    graph.delete_edges([(evader, del_neigh.index)])

    # step 2
    for _ in range(b - 1):
        try:
            broker = graph_metric.get_min(
                list(set(evader_neighbors) - set(del_neigh.neighbors()))
            )
        except Exception:
            raise StopIteration()
        graph.add_edge(del_neigh.index, broker.index)

    return graph


def get_roam_graphs(graph, boss, excecutions):

    def apply_with_b(graph, evader, b, executions):
        graphs = [graph]
        for _ in range(executions):
            try:
                graph = remove_one_add_many(graph, evader, b)
            except StopIteration:
                break
            graphs.append(graph)
        return graphs

    roam1 = apply_with_b(graph, boss, 1, excecutions)
    roam2 = apply_with_b(graph, boss, 2, excecutions)
    roam3 = apply_with_b(graph, boss, 3, excecutions)
    roam4 = apply_with_b(graph, boss, 4, excecutions)
    return roam1, roam2, roam3, roam4


def get_metrics_plot(roams, boss, metric, output_format='.pdf'):

    def get_metrics(node, graphs, metric):
        return [metric(graph).get_node_ranking(node) for graph in graphs]

    plt.figure()
    plt.title(metric.NAME)
    plt.plot(get_metrics(boss, roams[0], metric), label='roam1')
    plt.plot(get_metrics(boss, roams[1], metric), label='roam2')
    plt.plot(get_metrics(boss, roams[2], metric), label='roam3')
    plt.plot(get_metrics(boss, roams[3], metric), label='roam4')

    plt.legend()
    plt.savefig(metric.NAME + output_format)


def get_influence_value(roams, boss, influence, output_format='.pdf'):

    def get_metrics(node, graphs, influence):
        return [influence(graph).apply_metric(node) for graph in graphs]

    plt.figure()
    plt.title(influence.NAME)
    plt.plot(get_metrics(boss, roams[0], influence), label='roam1')
    plt.plot(get_metrics(boss, roams[1], influence), label='roam2')
    plt.plot(get_metrics(boss, roams[2], influence), label='roam3')
    plt.plot(get_metrics(boss, roams[3], influence), label='roam4')

    plt.legend()
    plt.savefig(influence.NAME + output_format)


def generate_metric_plots(graph, boss):
    roams = get_roam_graphs(graph, boss, 30)

    get_metrics_plot(roams, boss, DegreeMetric)
    get_metrics_plot(roams, boss, BetweennessMetric)
    get_metrics_plot(roams, boss, ClosenessMetric)
    get_metrics_plot(roams, boss, EigenVectorMetric)
    get_metrics_plot(roams, boss, SecondOrderDegreeMassMetric)
    get_influence_value(roams, boss, IndependentCascadeInfluence)
    get_influence_value(roams, boss, LinearThresholdInfluence)


graph = random_graph()
boss = find_the_boss(graph)
generate_metric_plots(graph, boss)
