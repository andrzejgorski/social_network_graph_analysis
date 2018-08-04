

def get_ranking_result(graph_sets, boss, metric_cls):
    return [
        [
            metric_cls(graph, boss).get_node_ranking(boss) for graph in g_set
        ]
        for g_set in graph_sets
    ]


def get_metric_values(graph_sets, boss, metric_cls):
    return [
        [
            metric_cls(graph, boss).apply_metric(boss) for graph in g_set
        ]
        for g_set in graph_sets
    ]
