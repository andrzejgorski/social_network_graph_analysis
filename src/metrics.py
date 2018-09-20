from igraph import Graph
from zope.dottedname import resolve
from heapq import nsmallest, nlargest
import numpy as np
from sklearn.preprocessing import normalize


class Metric(object):
    NAME = ''

    def __init__(self, graph, boss=None, *args, **kwargs):
        self.graph = graph
        self.boss = boss
        if self.graph:
            self._cache_metrics(*args, **kwargs)

    def apply_metric(self, node):
        raise NotImplementedError()

    def get_max(self, nodes=None):
        nodes = nodes or self.graph.vs
        return max(nodes, key=self.apply_metric)

    def get_min(self, nodes=None):
        nodes = nodes or self.graph.vs
        return min(nodes, key=self.apply_metric)

    def get_nmin(self, n, nodes):
        return nsmallest(n, nodes, key=lambda x: (self.apply_metric(x), x.index))

    def get_nmax(self, n, nodes):
        return nlargest(n, nodes, key=lambda x: (self.apply_metric(x), x.index))

    def get_sorted_nodes(self):
        values = self._calc_values()
        return sorted(self.graph.vs, key=lambda v: (values[v.index], v.index == self.boss), reverse=True)

    def get_node_ranking(self, node_index):
        node = self.graph.vs[node_index]
        return self.get_sorted_nodes().index(node)

    def _cache_metrics(self):
        raise NotImplementedError()

    def _calc_values(self):
        raise NotImplementedError()

    @classmethod
    def _get_index(cls, node):
        try:
            return node.index
        except AttributeError:
            return int(node)

    def _get_node(self, node):
        if type(node) == int:
            return self.graph.vs[node]
        return node

    @property
    def name(self):
        return self.NAME


class NodeMetric(Metric):

    def _apply_metric(self, node):
        raise NotImplementedError()

    def _cache_metrics(self):
        pass

    def _calc_values(self):
        return [self.apply_metric(node) for node in self.graph.vs]

    def apply_metric(self, node):
        return self._apply_metric(self._get_node(node))


class GraphMetric(Metric):

    def _cache_metrics(self, *args, **kwargs):
        self.metric_values = {
            node.index: value
            for node, value in zip(
                self.graph.vs, self._calc_values(*args, **kwargs)
            )
        }

    def _calc_values(self):
        raise NotImplementedError()

    def apply_metric(self, node):
        return self.metric_values[self._get_index(node)]


class DegreeMetric(NodeMetric):
    NAME = 'Degree'

    def _apply_metric(self, node):
        return node.degree()


class BetweennessMetric(NodeMetric):
    NAME = 'Betweenness'

    def _apply_metric(self, node):
        return node.betweenness()


class ClosenessMetric(NodeMetric):
    NAME = 'Closeness'

    def _apply_metric(self, node):
        return node.closeness()


class INGScoreMetric(GraphMetric):
    NAME = 'ING'

    def __init__(self, graph, boss, iterations=2,
                 benchmark_centrality=DegreeMetric,
                 linear_transformation=None,
                 *args, **kwargs):
        try:
            self.benchmark_centrality = benchmark_centrality(graph, boss, **kwargs)
        except:
            benchmark_centrality = resolve.resolve(benchmark_centrality)
            self.benchmark_centrality = benchmark_centrality(graph, boss, **kwargs)
        self.iterations = iterations
        self.linear_transformation = linear_transformation or self.get_adjacency
        self.kwargs = kwargs
        super(INGScoreMetric, self).__init__(graph, boss, *args, **kwargs)

    def _calc_values(self):
        matrix = np.array(self.linear_transformation(self.graph).data)

        s_vector = np.array(self.benchmark_centrality._calc_values())
        for _ in range(self.iterations):
            s_vector = np.dot(matrix, s_vector)
            s_vector = self.normalize(s_vector)
        return s_vector

    @property
    def name(self):
        return 'ING(c={}, k = {})'.format(
            self.benchmark_centrality.name, self.iterations
        )

    @classmethod
    def normalize(cls, vector):
        return normalize(np.array(vector)[:,np.newaxis], axis=0).ravel()

    @classmethod
    def get_adjacency(cls, graph):
        return graph.get_adjacency()


class KCoreDecompositionMetric(GraphMetric):
    NAME = 'k-core Decomposition'

    def _calc_values(self):
        return self.graph.shell_index()


class ExtendedKCoreDecompositionMetric(GraphMetric):
    NAME = 'Extended k-core Decomposition'

    def _calc_values(self):
        shell_index = self.graph.shell_index()
        degree = self.graph.degree()
        size = len(self.graph.vs)
        return [shell_index[i] + degree[i] / size for i in range(size)]


class NeighborhoodCorenessMetric(GraphMetric):
    NAME = 'Neighborhood Coreness'

    def _calc_values(self):
        shell_index = self.graph.shell_index()
        return [sum([shell_index[neighbor] for neighbor in neighbors])
                for neighbors in self.graph.neighborhood()]


class ExtendedNeighborhoodCorenessMetric(GraphMetric):
    NAME = 'Extended Neighborhood Coreness'

    def _calc_values(self):
        shell_index = self.graph.shell_index()
        neighborhood_coreness = [sum([shell_index[neighbor] for neighbor in neighbors])
                                 for neighbors in self.graph.neighborhood()]
        return [sum([neighborhood_coreness[neighbor] for neighbor in neighbors])
                for neighbors in self.graph.neighborhood()]


class SecondOrderDegreeMassMetric(NodeMetric):
    NAME = '2nd Order Degree Mass'

    def _apply_metric(self, node):
        first_degree_set = set(node.neighbors())
        first_degree_set.add(node.index)
        second_degree_set = set()
        for node in first_degree_set:
            second_degree_set |= set(self.graph.neighbors(node))
        return len(second_degree_set)


class EigenVectorMetric(GraphMetric):
    NAME = 'Eigenvector'

    def _calc_values(self):
        return self.graph.evcent()


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class ShapleyValueMetric(GraphMetric):
    def _calc_values(self):
        nodes_set = set(self.graph.vs)
        self._sub_values = {
            node: 0 for node in nodes_set
        }
        self._characteristic_values_cache = {}
        self._init_weights(len(nodes_set))
        self._calc_shapley_metric(nodes_set)

    def _calc_shapley_metric(self, nodes_set):
        raise NotImplementedError()

    def _init_weights(self, size):
        self._weight = []
        numerator = [1]
        difficulty = [size]
        for i in range(1, size + 1):
            numerator.append(numerator[i - 1] * i)
            difficulty.append(difficulty[i - 1] * (size - i))

        for i in range(0, size + 1):
            self._weight.append(float(numerator[i]) / difficulty[i])


class EffectivenessMetric(GraphMetric):
    NAME = 'Effectiveness'

    def __init__(self, graph, boss, step_numbers=1, *args, **kwargs):

        self.step_numbers = int(step_numbers)
        super(EffectivenessMetric, self).__init__(graph, boss, *args, **kwargs)

    @property
    def name(self):
        return (
            'Effectiveness(d = {})'
            .format(self.step_numbers)
        )

    def _calc_values(self):
        step_cache = {
            node: [(1.0, neighbor) for neighbor in node.neighbors()]
            for node in self.graph.vs
        }
        results = [0.0 for _ in self.graph.vs]
        for _ in range(self.step_numbers):
            next_cache = {}
            for node, pair in step_cache.items():
                next_cache[node] = []
                for coefficient, neighbor in pair:
                    new_coefficient = coefficient / neighbor.degree()
                    results[node.index] += new_coefficient
                    next_cache[node].extend(
                        (new_coefficient, second_neighbor)
                        for second_neighbor in neighbor.neighbors()
                    )

            step_cache = next_cache

        return results


class EffectivenessStep2Metric(EffectivenessMetric):
    def __init__(self, graph, boss, step_numbers=2, *args, **kwargs):
        self.step_numbers = int(step_numbers)
        super(EffectivenessMetric, self).__init__(graph, boss, *args, **kwargs)


class EffectivenessStep3Metric(EffectivenessMetric):
    def __init__(self, graph, boss, step_numbers=3, *args, **kwargs):
        self.step_numbers = int(step_numbers)
        super(EffectivenessMetric, self).__init__(graph, boss, *args, **kwargs)


class AtMost1DegreeAwayShapleyValue(GraphMetric):
    NAME = 'Coalition And Neighbors Shapley Value'

    def _calc_values(self):
        result = [self._marginal(node) for node in self.graph.vs]
        for node in self.graph.vs:
            for neighbor in node.neighbors():
                result[node.index] += self._marginal(neighbor)
        return result

    def _marginal(self, node):
        return 1.0 / (1 + node.degree())


class AtLeastKNeighborsInCoalitionShapleyValue(GraphMetric):
    NAME = 'At Least k Neighbors Shapley Value'

    def __init__(self, graph, boss, infection_factor=2, *args, **kwargs):
        self.infection_factor = float(infection_factor)
        super(AtLeastKNeighborsInCoalitionShapleyValue, self).__init__(
            graph, boss, *args, **kwargs)

    @property
    def name(self):
        return (
            'At Least {} Neighbors Shapley Value'.format(int(self.infection_factor))
        )

    def _calc_values(self, *args, **kwargs):
        result = [
            min(1, self.infection_factor / (1 + node.degree()))
            for node in self.graph.vs
        ]
        for node in self.graph.vs:
            for neighbor in node.neighbors():
                degree = neighbor.degree()
                result[node.index] += max(
                    0,
                    (degree - self.infection_factor + 1)
                    / (degree * (1 + degree))
                )
        return result


class NaiveShapleyValueMetric(ShapleyValueMetric):

    def _calc_shapley_metric(self, nodes_set):
        # Calculating Shapley Value for all vertices
        # Be carefull with using this function. The complexity of it is
        # O(2^n * n)
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

    def _characteristic_function(self, subset):
        raise NotImplementedError()

    def _get_characteristic_value(self, subset):
        if subset in self._characteristic_values_cache:
            return self._characteristic_values_cache[subset]
        value = self._characteristic_function(self, subset)
        self._characteristic_values_cache[subset] = value
        return value


class MetricCreator(object):
    def __init__(self, metric_class, *args, **kwargs):
        self.metric_class = metric_class
        self.kwargs = kwargs
        self.args = args

    def create_metric(self, graph, evader=None):
        return self.metric_class(graph, evader, *self.args, **self.kwargs)

    def __eq__(self, other):
        return (
            (self.metric_class, self.kwargs, self.args)
            == (other.metric_class, other.kwargs, other.args)
        )

    def __repr__(self):
        if self.args:
            args = ', args: {}'.format(self.args)
        else:
            args = ''
        if self.kwargs:
            kwargs = ', kwargs: {}'.format(self.kwargs)
        else:
            kwargs = ''
        return (
            'MetricCreator {}{}{}'
            .format(self.metric_class, args, kwargs)
        )

    @property
    def name(self):
        metric = self.metric_class(None, 0, *self.args, **self.kwargs)
        return metric.name

    def __call__(self, *args, **kwargs):
        return self.create_metric(*args, **kwargs)


class INGScoreMetricNC1(INGScoreMetric):
    NAME = 'ING'

    def __init__(self, graph, boss, iterations=1,
                 benchmark_centrality=NeighborhoodCorenessMetric,
                 linear_transformation=None,
                 *args, **kwargs):
        try:
            self.benchmark_centrality = benchmark_centrality(graph, boss, **kwargs)
        except:
            benchmark_centrality = resolve.resolve(benchmark_centrality)
            self.benchmark_centrality = benchmark_centrality(graph, boss, **kwargs)
        self.iterations = iterations
        self.linear_transformation = linear_transformation or self.get_adjacency
        self.kwargs = kwargs
        super(INGScoreMetric, self).__init__(graph, boss, *args, **kwargs)


class INGScoreMetricD1(INGScoreMetric):
    NAME = 'ING'

    def __init__(self, graph, boss, iterations=1,
                 benchmark_centrality=DegreeMetric,
                 linear_transformation=None,
                 *args, **kwargs):
        try:
            self.benchmark_centrality = benchmark_centrality(graph, boss, **kwargs)
        except:
            benchmark_centrality = resolve.resolve(benchmark_centrality)
            self.benchmark_centrality = benchmark_centrality(graph, boss, **kwargs)
        self.iterations = iterations
        self.linear_transformation = linear_transformation or self.get_adjacency
        self.kwargs = kwargs
        super(INGScoreMetric, self).__init__(graph, boss, *args, **kwargs)


class INGScoreMetricKC1(INGScoreMetric):
    NAME = 'ING'

    def __init__(self, graph, boss, iterations=1,
                 benchmark_centrality=KCoreDecompositionMetric,
                 linear_transformation=None,
                 *args, **kwargs):
        try:
            self.benchmark_centrality = benchmark_centrality(graph, boss, **kwargs)
        except:
            benchmark_centrality = resolve.resolve(benchmark_centrality)
            self.benchmark_centrality = benchmark_centrality(graph, boss, **kwargs)
        self.iterations = iterations
        self.linear_transformation = linear_transformation or self.get_adjacency
        self.kwargs = kwargs
        super(INGScoreMetric, self).__init__(graph, boss, *args, **kwargs)


SIMPLE_METRICS = [
    DegreeMetric,
    BetweennessMetric,
    ClosenessMetric,
    EigenVectorMetric,
    SecondOrderDegreeMassMetric,
    # KCoreDecompositionMetric,
    ExtendedKCoreDecompositionMetric,
    NeighborhoodCorenessMetric,
    ExtendedNeighborhoodCorenessMetric,
    AtMost1DegreeAwayShapleyValue,
    AtLeastKNeighborsInCoalitionShapleyValue,
    EffectivenessMetric,
    EffectivenessStep2Metric,
    EffectivenessStep3Metric,
    INGScoreMetricD1,
    INGScoreMetricKC1,
    INGScoreMetricNC1
]
