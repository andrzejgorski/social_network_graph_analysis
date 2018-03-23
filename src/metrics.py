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

    @classmethod
    def _get_index(cls, node):
        try:
            return node.index
        except AttributeError:
            return int(node)


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
        return self.metric_values[self._get_index(node)]


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


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class NaiveShapleyValueMetric(GraphMetric):
    def _calc_values(self):
        # Calculating Shapley Value for all vertices
        # Be carefull with using this function. The complexity of it is
        # O(2^n * n)
        nodes_set = set(self.graph.vs)
        self._sub_values = {
            node: 0 for node in nodes_set
        }
        self._characteristic_values_cache = {}
        self._init_weights(len(nodes_set))
        for subset in powerset(nodes_set):
            for vertex in nodes_set - subset:
                self._sub_values[vertex] -= (
                    self._get_characteristic_value(subset)
                    * self._weight[len(subset)]
                )

            for vertex in subset:
                self._sub_values[vertex] += (
                    self._get_characteristic_value(subset)
                    * self._weight[len(subset) - 1]
                )

    def _get_characteristic_value(self, subset):
        if subset in self._characteristic_values_cache:
            return self._characteristic_values_cache[subset]
        value = self._char_function(self, subset)
        self._characteristic_values_cache[subset] = value
        return value

    def _init_weights(self, size):
        self._weight = []
        numerator = [1]
        difficulty = [size]
        for i in range(1, size + 1):
            numerator.append(numerator[i - 1] * i)
            difficulty.append(difficulty[i - 1] * (size - i))

        for i in range(0, size + 1):
            self._weight.append(float(numerator[i]) / difficulty[i])
