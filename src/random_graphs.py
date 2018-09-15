import igraph
from datetime import datetime


def generate_random_graphs(samples=None, nodes=None, algorithm=None, **kwargs):
    for i in range(samples):
        # print(str(datetime.now().time()) + ": Generating sample " + str(i+1))
        yield algorithm(nodes, **kwargs)


def scale_free_network(nodes, links=None, **kwargs):
    return igraph.Graph.Barabasi(nodes, links)


def small_world_network(nodes, average_degree=None, rewiring_probability=None, **kwargs):
    return igraph.Graph.Watts_Strogatz(1, nodes, average_degree // 2, rewiring_probability)


def Erdos_Renyi_graph(nodes, average_degree=None, **kwargs):
    return igraph.Graph.Erdos_Renyi(nodes, m=int(nodes * average_degree / 2.0))
