from metrics import DegreeMetric


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
        brokers = graph_metric.get_nmin(
            b - 1, list(set(evader_neighbors) - set(del_neigh.neighbors())))
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
    evader_neighbors.remove(del_neigh)
    graph.es.select(_source=del_neigh.index).delete()
    graph.es.select(_target=del_neigh.index).delete()

    # step 2
    try:
        bot_neighbors = graph_metric.get_nmax(b - 1, evader_neighbors)
    except Exception:
        return graph
    graph.add_edges(
        [(del_neigh.index, neighbor.index) for neighbor in bot_neighbors])

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
