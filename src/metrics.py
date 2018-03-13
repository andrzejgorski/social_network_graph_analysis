import random
from functools import partial


class Metric(object):
    NAME = ''

    def __init__(self, graph, *args, **kwargs):
        self.graph = graph
        self._cache_metrics(*args, **kwargs)

    def apply_metric(self, node):
        raise NotImplementedError()

    def get_max(self, nodes):
        return max(nodes, key=self.apply_metric)

    def get_min(self, nodes):
        return min(nodes, key=self.apply_metric)

    def get_sorted_nodes(self):
        return sorted(self.graph.vs, key=self.apply_metric, reverse=True)

    def get_node_ranking(self, node_index):
        node = self.graph.vs[node_index]
        return self.get_sorted_nodes().index(node)

    def _cache_metrics(self):
        raise NotImplementedError()


class NodeMetric(Metric):

    def _cache_metrics(self):
        pass


class GraphMetric(Metric):

    def _cache_metrics(self):
        self.metric_values = {
            node.index: value
            for node, value in zip(self.graph.vs, self._calc_values())
        }

    def _calc_values(self):
        raise NotImplementedError()

    def apply_metric(self, node):
        return self.metric_values[node.index]


class DegreeMetric(NodeMetric):
    NAME = 'degree'

    def apply_metric(self, node):
        return node.degree()


class BetweennessMetric(NodeMetric):
    NAME = 'betweenness'

    def apply_metric(self, node):
        return node.betweenness()


class ClosenessMetric(NodeMetric):
    NAME = 'closeness'

    def apply_metric(self, node):
        return node.closeness()


class SecondOrderDegreeMassMetric(NodeMetric):
    NAME = '2nd order degree mass'

    def apply_metric(self, node):
        first_degree_set = set(node.neighbors())
        first_degree_set.add(node.index)
        second_degree_set = set()
        for node in first_degree_set:
            second_degree_set |= set(self.graph.neighbors(node))
        return len(second_degree_set)


class EigenVectorMetric(GraphMetric):
    NAME = 'eigenvector'

    def _calc_values(self):
        return self.graph.evcent()
