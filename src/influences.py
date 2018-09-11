import random
from metrics import NodeMetric


class MonteCarloSamplingInfluence(NodeMetric):

    def __init__(self, graph, boss, samplings=None, dummy_alpha=None, *args, **kwargs):
        self.samplings = samplings
        self.dummy_alpha = dummy_alpha
        super(MonteCarloSamplingInfluence, self).__init__(
            graph, boss, *args, **kwargs)

    def _apply_metric(self, node, *args, **kwargs):
        self.subvalues = [0 for _ in self.graph.vs]
        for _ in range(self.samplings):
            samplings = self._sampling_function(node, *args, **kwargs)
            for node_index in samplings:
                self.subvalues[node_index] += 1

        dummy_beginning = len(self.subvalues)
        for v in self.graph.vs:
            if v.attributes() and v.attributes()['dummy']:
                dummy_beginning = v.index
                break
        self.subvalues = self.subvalues[:dummy_beginning]

        return sum([float(value)/self.samplings for value in self.subvalues])/len(self.subvalues)

    def _sampling_function(self, node):
        raise NotImplementedError()


class IndependentCascadeInfluence(MonteCarloSamplingInfluence):
    NAME = 'independent_cascade'

    def _sampling_function(self, node, mean_influence=0.5, *args, **kwargs):
        recently_activated = {node.index}

        for v in self.graph.vs:
            if not v.attributes():
                break
            if v.attributes()['dummy']:
                recently_activated.add(v.index)

        active = recently_activated.copy()

        graph = self.graph.copy()
        graph.es["weight"] = 1.0
        for edge in graph.es:
            graph[edge.tuple[0], edge.tuple[1]] = random.uniform(0, 1)

        while recently_activated:
            new_iteration = set()
            for v in recently_activated:
                not_active_neighbors = set(graph.neighbors(v)) - active
                for neighbor in not_active_neighbors:
                    edge_influence = mean_influence
                    if graph.vs[v].attributes() and graph.vs[v].attributes()['dummy']:
                        edge_influence *= self.dummy_alpha
                    if graph[v, neighbor] < edge_influence:
                        new_iteration.add(neighbor)
                        active.add(neighbor)
            recently_activated = new_iteration

        return active


class LinearThresholdInfluence(MonteCarloSamplingInfluence):
    NAME = 'linear_threshold'

    def _sampling_function(self, node, *args, **kwargs):
        recently_activated = {node.index}

        for v in self.graph.vs:
            if not v.attributes():
                break
            if v.attributes()['dummy']:
                recently_activated.add(v.index)

        active = recently_activated.copy()

        inactive = set(range(0, self.graph.vcount())) - active

        thresholds = [
            random.randint(1, max(1, self.graph.vs[i].degree()))
            for i in range(0, self.graph.vcount())
        ]

        while recently_activated:
            new_iteration = set()
            for v in inactive:
                active_neighbors = set(self.graph.neighbors(v)) & active
                neighbors_to_remove = set()
                for neighbor in active_neighbors:
                    if self.graph.vs[neighbor].attributes() and self.graph.vs[neighbor].attributes()['dummy']:
                        if random.uniform(0, 1) > self.dummy_alpha:
                            neighbors_to_remove.add(neighbor)
                active_neighbors -= neighbors_to_remove
                if thresholds[v] <= len(active_neighbors):
                    new_iteration.add(v)
            active |= new_iteration
            inactive -= new_iteration
            recently_activated = new_iteration

        return active
