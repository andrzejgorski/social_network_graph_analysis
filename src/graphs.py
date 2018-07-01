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
)


def random_graph(nodes=20):
    return Graph.GRG(nodes, 0.5)


def find_the_boss(graph):
    return graph.degree().index(max(graph.degree()))


def remove_one_add_many(graph, evader, b, metric):
    graph = graph.copy()

    # step 1
    evader_neighbors = graph.vs[evader].neighbors()
    if len(evader_neighbors) == 0:
        raise StopIteration()

    graph_metric = metric(graph, evader)
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


def save_metric_ranking_plot(roams, boss, metric_cls, output_format='.pdf', **kwargs):
    def get_metrics(node, graphs, metric):
        return [metric_cls(graph, node, **kwargs).get_node_ranking(node)
                for graph in graphs]

    plt.figure()
    metric = metric_cls(graph, boss, **kwargs)
    plt.title(metric.NAME)

    plt.plot(get_metrics(boss, roams[0], metric_cls), label='roam1')
    # plt.plot(get_metrics(boss, roams[4], metric), label='roam1eig')
    plt.plot(get_metrics(boss, roams[1], metric_cls), label='roam2')
    # plt.plot(get_metrics(boss, roams[5], metric), label='roam2eig')
    plt.plot(get_metrics(boss, roams[2], metric_cls), label='roam3')
    # plt.plot(get_metrics(boss, roams[6], metric), label='roam3eig')
    plt.plot(get_metrics(boss, roams[3], metric_cls), label='roam4')
    # plt.plot(get_metrics(boss, roams[7], metric), label='roam4eig')

    plt.legend(loc=2)
    plt.xlabel("iterations")
    plt.ylabel("ranking")
    plt.savefig(metric.NAME + output_format)


def generate_metric_plots(graph, boss):
    roams = get_roam_graphs(graph, boss, 4, metric=DegreeMetric)
    # roams += get_roam_graphs(graph, boss, 4, metric=SecondOrderDegreeMassMetric)

    # save_metric_ranking_plot(roams, boss, DegreeMetric)
    # save_metric_ranking_plot(roams, boss, BetweennessMetric)
    # save_metric_ranking_plot(roams, boss, ClosenessMetric)
    # save_metric_ranking_plot(roams, boss, EigenVectorMetric)
    # save_metric_ranking_plot(roams, boss, SecondOrderDegreeMassMetric)
    # save_metric_ranking_plot(roams, boss, KCoreDecompositionMetric)
    # save_metric_ranking_plot(roams, boss, ExtendedKCoreDecompositionMetric)
    # save_metric_ranking_plot(roams, boss, NeighborhoodCorenessMetric)
    # save_metric_ranking_plot(roams, boss, ExtendedNeighborhoodCorenessMetric)
    # save_metric_ranking_plot(roams, boss, AtMost1DegreeAwayShapleyValue)
    # save_metric_ranking_plot(roams, boss, AtLeastKNeighborsInCoalitionShapleyValue)
    # save_metric_ranking_plot(roams, boss, IndependentCascadeInfluence)
    # save_metric_ranking_plot(roams, boss, LinearThresholdInfluence)
    # save_metric_ranking_plot(roams, boss, EffectivenessMetric)
    # save_metric_ranking_plot(roams, boss, EffectivenessMetric, step_numbers=2)
    # save_metric_ranking_plot(roams, boss, EffectivenessMetric, step_numbers=3)


graph = random_graph(nodes=20)
boss = find_the_boss(graph)
generate_metric_plots(graph, boss)
