from scipy import stats
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


def add_bot_assistant(graph, evader, b, metric):
    graph = graph.copy()

    # adding assistant
    graph.add_vertex()
    assistant = graph.vs[graph.vcount()-1]
    graph.add_edges([(evader, assistant.index)])

    for _ in range(b):
        evader_neighbors = graph.vs[evader].neighbors()
        if not evader_neighbors:
            raise StopIteration()
        graph_metric = metric(graph)
        neighbour = graph_metric.get_max(evader_neighbors)
        graph.delete_edges([(evader, neighbour.index)])

        graph.add_edges(
            [(assistant.index, neighbour.index)]
        )

    return graph


def get_cut_graphs(graph, boss, executions, function=remove_one_add_many,
                   metric=DegreeMetric):

    def apply_with_b(graph, evader, b, executions):
        graphs = [graph]
        for _ in range(executions):
            try:
                graph = function(graph, evader, b, metric)
            except StopIteration:
                break
            graphs.append(graph)
        return graphs

    graph1 = apply_with_b(graph, boss, 1, executions)
    graph2 = apply_with_b(graph, boss, 2, executions)
    graph3 = apply_with_b(graph, boss, 3, executions)
    graph4 = apply_with_b(graph, boss, 4, executions)
    return graph1, graph2, graph3, graph4


def get_ranking_result(graph_sets, boss, metric_cls):
    return [
        [
            metric_cls(graph, boss).get_node_ranking(boss) for graph in g_set
        ]
        for g_set in graph_sets
    ]


def get_ranking_scores(ranking_results, metric_name=None):
    scores = []
    shifted_scores = []

    for result in ranking_results:
        scores.append(calculate_integral_score(result))
        shifted_scores.append(calculate_relative_integral_score(result))

    scores.append(sum(scores) / float(len(scores)))
    if metric_name:
        scores.insert(0, metric_name)
    shifted_scores.append(sum(shifted_scores) / float(len(shifted_scores)))
    if metric_name:
        shifted_scores.insert(0, metric_name)

    return scores, shifted_scores


def save_metric_ranking_plot(results, metric_name, label, output_file=None):
    output_format = '.jpeg'

    fig, ax = plt.subplots()
    plt.title(metric_name)
    colors = ('purple', 'green', 'r', 'c', 'm', 'y', 'k', 'w')
    shapes = ('s', '^', 'o', 'v', 'D', 'p', 'x', '8')
    linestyles = ((0, (15, 10, 3, 10)), '--', ':', '-.')

    for i in range(len(results)):
        label_index = label + str(i + 1)
        line = plt.plot(list(map(lambda x: x + 1, results[i])), label=label_index)
        plt.setp(line, marker=shapes[i], markersize=15.0, markeredgewidth=2, markerfacecolor="None",
                 markeredgecolor=colors[i], linewidth=2, linestyle=linestyles[i], color=colors[i])

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().invert_yaxis()
    plt.legend(loc='lower left')
    plt.margins(0.1)
    plt.xlabel("iterations")
    plt.ylabel("ranking")

    output_file = output_file or metric_name + output_format

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def save_metric_ranking_plot_for_random_graphs(results, metric_name, label, output_file=None):
    output_format = '.jpeg'

    plt.subplots()
    plt.title(metric_name)
    colors = ('purple', 'green', 'r', 'c', 'm', 'y', 'k', 'w')
    shapes = ('s', '^', 'o', 'v', 'D', 'p', 'x', '8')
    linestyles = ((0, (15, 10, 3, 10)), '--', ':', '-.')

    for i in range(len(results)):
        label_index = label + str(i + 1)
        line = plt.plot(list(map(lambda x: x[0] + 1, results[i])), label=label_index)
        plt.setp(line, marker=shapes[i], markersize=15.0, markeredgewidth=2, markerfacecolor="None",
                 markeredgecolor=colors[i], linewidth=2, linestyle=linestyles[i], color=colors[i])
        plt.fill_between(range(len(results[i])),
                         list(map(lambda x: x[1][0] + 1, results[i])),
                         list(map(lambda x: x[1][1] + 1, results[i])),
                         facecolor=colors[i], edgecolors=None,
                         alpha=0.2)

    plt.gca().invert_yaxis()
    plt.legend(loc='lower left')
    plt.margins(0.1)
    plt.xlabel("iterations")
    plt.ylabel("ranking")

    output_file = output_file or metric_name + output_format

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def get_metric_values(graph_sets, boss, metric_cls):
    return [
        [
            metric_cls(graph, boss).apply_metric(boss) for graph in g_set
        ]
        for g_set in graph_sets
    ]


def save_influence_value_plot(metric_values, metric_name, label,
                              output_file=None):
    output_format='.jpeg'

    plt.subplots()
    plt.title(metric_name)
    colors = ('purple', 'green', 'r', 'c', 'm', 'y', 'k', 'w')
    shapes = ('s', '^', 'o', 'v', 'D', 'p', 'x', '8')
    linestyles = ((0, (15, 10, 3, 10)), '--', ':', '-.')

    for i in range(len(metric_values)):
        label_index = label + str(i + 1)
        line = plt.plot(list(map(lambda x: x + 1, metric_values[i])), label=label_index)
        plt.setp(line, marker=shapes[i], markersize=15.0, markeredgewidth=2, markerfacecolor="None",
                 markeredgecolor=colors[i], linewidth=2, linestyle=linestyles[i], color=colors[i])

    plt.legend(loc='lower left')
    plt.margins(0.1)
    plt.xlabel("iterations")
    plt.ylabel("value")
    output_file = output_file or metric_name + output_format
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def save_scores_table(scores_table, label, output_file='scores_table.pdf'):
    sorted_scores = sorted(scores_table, key=lambda score: score[5])

    plt.figure()
    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=sorted_scores,
             colLabels=('METRIC name', label + '(1)', label + '(2)', label + '(3)', label + '(4)', 'AVERAGE'),
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
