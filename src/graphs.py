from igraph import Graph
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
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
    calculate_integral_score,
    calculate_relative_integral_score,
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


def remove_one_bot_enters(graph, evader, b, metric):
    graph = graph.copy()

    # step 1
    evader_neighbors = graph.vs[evader].neighbors()
    if len(evader_neighbors) == 0:
        raise StopIteration()

    graph_metric = metric(graph)
    del_neigh = graph_metric.get_min(evader_neighbors)
    graph.delete_vertices([del_neigh.index])
    evader_neighbors.remove(del_neigh)

    # step 2
    graph.add_vertex()
    bot = graph.vs[graph.vcount()-1]
    try:
        bot_neighbors = graph_metric.get_nmax(b - 1, evader_neighbors)
    except Exception:
        return graph
    graph.add_edges([(bot.index, neighbor.index) for neighbor in bot_neighbors])

    return graph


class GraphList(list):
    def __init__(self, label=None, *args, **kwargs):
        super().__init__(args, **kwargs)
        self.label = label or 'graph_list'


def get_cut_graphs(graph, boss, excecutions, function=remove_one_add_many,
                   metric=DegreeMetric, label=None):

    def apply_with_b(graph, evader, b, executions):
        graphs = [graph]
        for _ in range(executions):
            try:
                graph = function(graph, evader, b, metric)
            except StopIteration:
                break
            graphs.append(graph)
        return graphs

    graph1 = apply_with_b(graph, boss, 1, excecutions)
    graph2 = apply_with_b(graph, boss, 2, excecutions)
    graph3 = apply_with_b(graph, boss, 3, excecutions)
    graph4 = apply_with_b(graph, boss, 4, excecutions)
    return GraphList(label, graph1, graph2, graph3, graph4)


def save_metric_ranking_plot(graph_sets, boss, metric_cls, output_file=None):
    output_format = '.jpeg'

    def get_metrics(node, graphs, metric):
        return [metric_cls(graph, boss).get_node_ranking(node)
                for graph in graphs]

    fig, ax = plt.subplots()
    graph = graph_sets[0][0]
    metric = metric_cls(graph, boss)
    plt.title(metric.name)
    colors = ('purple', 'green', 'r', 'c', 'm', 'y', 'k', 'w')
    shapes = ('s', '^', 'o', 'v', 'D', 'p', 'x', '8')
    linestyles = ((0, (15, 10, 3, 10)), '--', ':', '-.')
    results = [get_metrics(boss, g_set, metric_cls) for g_set in graph_sets]

    scores = []
    shifted_scores = []

    for i in range(len(results)):
        label = graph_sets.label + str(i + 1)
        # print("Integral score: {}, {}: {}".format(metric.name, label, calculate_integral_score(results[i])))
        scores.append(calculate_integral_score(results[i]))
        shifted_scores.append(calculate_relative_integral_score(results[i]))
        line = plt.plot(list(map(lambda x: x + 1, results[i])), label=label)
        plt.setp(line, marker=shapes[i], markersize=15.0, markeredgewidth=2, markerfacecolor="None",
                 markeredgecolor=colors[i], linewidth=2, linestyle=linestyles[i], color=colors[i])

    scores.append(sum(scores) / float(len(scores)))
    scores.insert(0, metric.name)

    shifted_scores.append(sum(shifted_scores) / float(len(shifted_scores)))
    shifted_scores.insert(0, metric.name)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().invert_yaxis()
    plt.legend(loc='lower left')
    plt.margins(0.1)
    plt.xlabel("iterations")
    plt.ylabel("ranking")

    output_file = output_file or metric.name + output_format

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    return scores, shifted_scores


def save_influence_value_plot(graph_set, boss, metric_cls, output_file=None,
                             **kwargs):
    output_format='.jpeg'

    def get_metrics(node, graphs, metric):
        return [metric_cls(graph, boss, **kwargs).apply_metric(node)
                for graph in graphs]

    plt.subplots()
    graph = graph_set[0][0]
    metric = metric_cls(graph, boss, **kwargs)
    plt.title(metric.name)
    colors = ('purple', 'green', 'r', 'c', 'm', 'y', 'k', 'w')
    shapes = ('s', '^', 'o', 'v', 'D', 'p', 'x', '8')
    linestyles = ((0, (15, 10, 3, 10)), '--', ':', '-.')
    results = [get_metrics(boss, g_set, metric_cls) for g_set in graph_set]

    for i in range(len(results)):
        label = 'robe' + str(i + 1)
        line = plt.plot(list(map(lambda x: x + 1, results[i])), label=label)
        plt.setp(line, marker=shapes[i], markersize=15.0, markeredgewidth=2, markerfacecolor="None",
                 markeredgecolor=colors[i], linewidth=2, linestyle=linestyles[i], color=colors[i])

    plt.legend(loc='lower left')
    plt.margins(0.1)
    plt.xlabel("iterations")
    plt.ylabel("value")
    output_file = output_file or metric.name + output_format
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def save_scores_table(scores_table, output_file='scores_table.pdf'):
    sorted_scores = sorted(scores_table, key=lambda score: score[5])

    plt.figure()
    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=sorted_scores,
             colLabels=('METRIC name', 'ROAM(1)', 'ROAM(2)', 'ROAM(3)', 'ROAM(4)', 'AVERAGE'),
             colWidths=[0.5] + [0.1] * 5,
             loc='upper center')
    fig.savefig(output_file)
    plt.close()


def get_influence_value(graph_set, boss, influence, output_format='.jpeg'):

    def get_metrics(node, graphs, influence):
        return [influence(graph, samplings=30000).apply_metric(node) for graph in graphs]

    plt.figure()
    plt.title(influence.name)

    plt.plot(get_metrics(boss, graph_set[0], influence), label=graphs_set.label + '1')
    plt.plot(get_metrics(boss, graph_set[1], influence), label=graphs_set.label + '2')
    plt.plot(get_metrics(boss, graph_set[2], influence), label=graphs_set.label + '3')
    plt.plot(get_metrics(boss, graph_set[3], influence), label=graphs_set.label + '4')

    plt.legend(loc=3)
    plt.xlabel("iterations")
    plt.ylabel("value")
    plt.savefig(influence.name + output_format)


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
