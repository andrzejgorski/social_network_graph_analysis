import igraph


def generate_random_graphs(samples=50, nodes=50, algorithm=None, **kwargs):
    for _ in range(samples):
        yield algorithm(nodes, **kwargs)


def scale_free_network(nodes, links=None):
    return igraph.Graph.Barabasi(nodes, links)


def small_world_network(nodes, average_degree=None, rewiring_probability=None):
    return igraph.Graph.Watts_Strogatz(1, nodes, average_degree // 2, rewiring_probability)


def Erdos_Renyi_graph(nodes, average_degree=None):
    return igraph.Graph.Erdos_Renyi(nodes, m=int(nodes * average_degree / 2.0))
