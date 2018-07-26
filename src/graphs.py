import matplotlib.pyplot as plt
from igraph import Graph
from matplotlib.ticker import MaxNLocator

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

scores_table = []


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
    fig, ax = plt.subplots()
    metric = metric_cls(graph, boss, **kwargs)
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
    scores_table.append(scores)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc=2)
    plt.margins(0.1)
    plt.xlabel("iterations")
    plt.ylabel("ranking")
    plt.savefig(metric.NAME + output_format, bbox_inches='tight')
    plt.close()


def save_scores_table(output_format='.pdf'):
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
    fig.savefig('scores_table' + output_format)
    plt.close()


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


graph = random_graph(nodes=20)
boss = find_the_boss(graph)
generate_metric_plots(graph, boss)
