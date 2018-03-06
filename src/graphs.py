from igraph import Graph
from functools import partial
from random import choice

import matplotlib.pyplot as plt
from igraph import Graph
from influences import independent_cascade, linear_threshold

from metrics import (
    DegreeMetric,
    BetweennessMetric,
    ClosenessMetric,
    EigenVectorMetric,
)


def random_graph(nodes=20):
    return Graph.GRG(nodes, 0.5)


def find_the_boss(graph):
    return graph.evcent().index(1)


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
    for _ in range(b - 1):
        try:
            broker = graph_metric.get_min(
                list(set(evader_neighbors) - set(del_neigh.neighbors()))
            )
        except Exception:
            raise StopIteration()
        graph.add_edge(del_neigh.index, broker.index)

    return graph


def get_roam_graphs(graph, boss, excecutions, metric):

    def apply_with_b(graph, evader, b, executions, metric):
        graphs = [graph]
        for _ in range(executions):
            try:
                graph = remove_one_add_many(graph, evader, b, metric)
            except StopIteration:
                break
            graphs.append(graph)
        return graphs

    roam1 = apply_with_b(graph, boss, 1, excecutions, metric)
    roam2 = apply_with_b(graph, boss, 2, excecutions, metric)
    roam3 = apply_with_b(graph, boss, 3, excecutions, metric)
    roam4 = apply_with_b(graph, boss, 4, excecutions, metric)
    return roam1, roam2, roam3, roam4


def apply_metrics(graph, boss, metric):
    roams = get_roam_graphs(graph, boss, 30, metric)

    def get_metrics(node, graphs, metric):
        return [metric(graph).get_node_ranking(node) for graph in graphs]

    plt.figure()
    plt.title(metric.NAME)
    plt.plot(get_metrics(boss, roams[0], metric), label='roam1')
    plt.plot(get_metrics(boss, roams[1], metric), label='roam2')
    plt.plot(get_metrics(boss, roams[2], metric), label='roam3')
    plt.plot(get_metrics(boss, roams[3], metric), label='roam4')

    plt.legend()
    plt.savefig(metric.NAME + '.pdf')


def generate_metric_plots(graph, boss):
    apply_metrics(graph, boss, DegreeMetric)
    apply_metrics(graph, boss, BetweennessMetric)
    apply_metrics(graph, boss, ClosenessMetric)
    apply_metrics(graph, boss, EigenVectorMetric)


graph = random_graph()
boss = find_the_boss(graph)
generate_metric_plots(graph, boss)
print(independent_cascade(graph, boss))
print(linear_threshold(graph, boss))
