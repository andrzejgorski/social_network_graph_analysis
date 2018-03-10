import random
from metrics import GraphMetric


class MonteCarloSamplingInfluence(GraphMetric):

    def __init__(self, graph, samplings=1000, *args, **kwargs):
        self.samplings = samplings
        super(MonteCarloSamplingInfluence, self).__init__(
            graph, *args, **kwargs)

    def _calc_values(self, *args, **kwargs):
        self.subvalues = [0 for _ in self.graph.vs]
        for _ in range(self.samplings):
            samplings = self._sampling_function(*args, **kwargs)
            for node in samplings:
                self.subvalues[node] += 1

        return [float(value)/self.samplings for value in self.subvalues]

    def _sampling_function(self):
        raise NotImplementedError()


class IndependentCascadeInfluence(MonteCarloSamplingInfluence):
    NAME = 'independent_cascade'

    def _sampling_function(self, mean_influence=0.2, *args, **kwargs):
        seed = random.choice(self.graph.vs)
        recently_activated = {seed.index}
        active = {seed.index}

        graph = self.graph.copy()
        graph.es["weight"] = 1.0
        for edge in graph.es:
            graph[edge.tuple[0], edge.tuple[1]] = random.uniform(0, 1)

        while recently_activated:
            new_iteration = set()
            for v in recently_activated:
                not_active_neighbors = set(graph.neighbors(v)) - active
                for neighbor in not_active_neighbors:
                    if graph[v, neighbor] < mean_influence:
                        new_iteration.add(neighbor)
                        active.add(neighbor)
            recently_activated = new_iteration

        return active


class LinearThresholdInfluence(MonteCarloSamplingInfluence):
    NAME = 'linear_threshold'

    def _sampling_function(self, *args, **kwargs):
        seed = random.choice(self.graph.vs)

        recently_activated = {seed.index}
        active = {seed.index}

        inactive = set(range(0, self.graph.vcount())) - active

        thresholds = [
            random.randint(1, self.graph.vcount())
            for _ in range(0, self.graph.vcount())
        ]

        while recently_activated:
            new_iteration = set()
            for v in inactive:
                active_neighbors = set(self.graph.neighbors(v)) & active
                if thresholds[v] <= len(active_neighbors):
                    new_iteration.add(v)
            active |= new_iteration
            inactive -= new_iteration
            recently_activated = new_iteration

        return active
