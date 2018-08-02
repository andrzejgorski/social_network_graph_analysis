from igraph import Graph
from matplotlib.ticker import MaxNLocator
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
    evader_neighbors.remove(del_neigh)

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


def save_metric_ranking_plot(roams, boss, metric_cls, output_file=None):
    output_format = '.jpeg'

    def get_metrics(node, graphs, metric):
        return [metric_cls(graph, boss).get_node_ranking(node)
                for graph in graphs]

    plt.figure()
    fig, ax = plt.subplots()
    graph = roams[0][0]
    metric = metric_cls(graph, boss)
    plt.title(metric.NAME)
    colors = ('purple', 'green', 'r', 'c', 'm', 'y', 'k', 'w')
    shapes = ('s', '^', 'o', 'v', 'D', 'p', 'x', '8')
    linestyles = ((0, (15, 10, 3, 10)), '--', ':', '-.')
    results = [get_metrics(boss, roam, metric_cls) for roam in roams]

    scores = []

    for i in range(len(results)):
        label = 'roam' + str(i + 1)
        # print("Integral score: {}, {}: {}".format(metric.NAME, label, calculate_integral_score(results[i])))
        scores.append(calculate_integral_score(results[i]))
        line = plt.plot(list(map(lambda x: x + 1, results[i])), label=label)
        plt.setp(line, marker=shapes[i], markersize=15.0, markeredgewidth=2, markerfacecolor="None",
                 markeredgecolor=colors[i], linewidth=2, linestyle=linestyles[i], color=colors[i])

    scores.append(sum(scores) / float(len(scores)))
    scores.insert(0, metric.NAME)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().invert_yaxis()
    plt.legend(loc='lower left')
    plt.margins(0.1)
    plt.xlabel("iterations")
    plt.ylabel("ranking")

    output_file = output_file or metric.NAME + output_format

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    return scores


def save_scores_table(scores_table, output_file='scores_table.pdf'):
    sorted_scores = sorted(scores_table, key=lambda score: score[5])

    plt.figure()
    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=sorted_scores,
             colLabels=('METRIC NAME', 'ROAM(1)', 'ROAM(2)', 'ROAM(3)', 'ROAM(4)', 'AVERAGE'),
             colWidths=[0.5] + [0.1] * 5,
             loc='upper center')
    fig.savefig(output_file)
    plt.close()


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
    save_metric_ranking_plot(roams, boss, EffectivenessMetric)
    save_metric_ranking_plot(roams, boss, EffectivenessMetric, step_numbers=2)
    save_metric_ranking_plot(roams, boss, EffectivenessMetric, step_numbers=3)
    save_metric_ranking_plot(roams, boss, INGScoreMetric, benchmark_centrality=KCoreDecompositionMetric, iterations=1, linear_transformation=INGScoreMetric.get_adjacency)
    save_metric_ranking_plot(roams, boss, INGScoreMetric, benchmark_centrality=NeighborhoodCorenessMetric, iterations=1, linear_transformation=INGScoreMetric.get_adjacency)
    save_metric_ranking_plot(roams, boss, INGScoreMetric, benchmark_centrality=DegreeMetric, iterations=1, linear_transformation=INGScoreMetric.get_adjacency)
    save_scores_table()
