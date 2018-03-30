from functools import partial
from igraph import Graph
from metrics import (
    DegreeMetric,
    BetweennessMetric,
    ClosenessMetric,
    EigenVectorMetric,
    SecondOrderDegreeMassMetric,
    AtMost1DegreeAwayShapleyValue,
    AtMostKDegreeAwayShapleyValue,
)


def get_4_elements_list_graph():
    return Graph(edges=[(0, 1), (1, 2), (2, 3)])


def get_4_free_nodes():
    return Graph(n=4)


def get_4_elements_clique():
    return Graph(edges=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def get_11_elements_star():
    return Graph(edges=[
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (0, 6), (0, 7), (0, 8), (0, 9), (0, 10)
    ])


class TestMetric(object):
    METRIC = DegreeMetric

    TEST_CASES = ()

    def __init__(self):
        for case in self.TEST_CASES:
            self.test_one(case[0], case[1])

    def test_one(self, get_graph, expected_results):
        graph = get_graph()
        metric = self.METRIC(graph)
        for i in range(len(graph.vs)):
            value = metric.apply_metric(i)
            self._assert_3_digits_eq(value, expected_results[i], (
                'test_{}_{} index: {}, value: {}, expected: {}'
                .format(
                    metric.NAME,
                    get_graph.__name__,
                    i,
                    value,
                    expected_results[i]
                ))
            )

    def _assert_3_digits_eq(self, value, expected_value, message):
        diff = value - expected_value
        assert abs(diff) <= 0.001, message


class TestsDegreeMetric(TestMetric):
    METRIC = DegreeMetric

    TEST_CASES = (
        (get_4_free_nodes, [0, 0, 0, 0]),
        (get_4_elements_list_graph, [1, 2, 2, 1]),
        (get_4_elements_clique, [3, 3, 3, 3]),
        (get_11_elements_star, [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )


class TestsBetweennessMetric(TestMetric):
    METRIC = BetweennessMetric

    TEST_CASES = (
        (get_4_free_nodes, [0, 0, 0, 0]),
        (get_4_elements_list_graph, [0, 2, 2, 0]),
        (get_4_elements_clique, [0, 0, 0, 0]),
        (get_11_elements_star, [45 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    )


class TestsClosenessMetric(TestMetric):
    METRIC = ClosenessMetric

    # TODO Check values
    TEST_CASES = (
        (get_4_free_nodes, [0.25, 0.25, 0.25, 0.25]),
        (get_4_elements_list_graph, [0.5, 0.75, 0.75, 0.5]),
        (get_4_elements_clique, [1, 1, 1, 1]),
        (get_11_elements_star, [
            1, 0.526, 0.526, 0.526, 0.526, 0.526,
            0.526, 0.526, 0.526, 0.526, 0.526
        ]),
    )


class TestsEigenVectorMetric(TestMetric):
    METRIC = EigenVectorMetric

    # TODO Check values
    TEST_CASES = (
        (get_4_free_nodes, [1, 1, 1, 1]),
        (get_4_elements_list_graph, [0.618, 1, 1, 0.618]),
        (get_4_elements_clique, [1, 1, 1, 1]),
        (get_11_elements_star, [
            1 , 0.316, 0.316, 0.316, 0.316, 0.316,
            0.316, 0.316, 0.316, 0.316, 0.316
        ]),
    )


class TestsSecondOrderDegreeMassMetric(TestMetric):
    METRIC = SecondOrderDegreeMassMetric

    TEST_CASES = (
        (get_4_free_nodes, [0, 0, 0, 0]),
        (get_4_elements_list_graph, [3, 4, 4, 3]),
        (get_4_elements_clique, [4, 4, 4, 4]),
        (get_11_elements_star, [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]),
    )


class TestsAtMost1DegreeAwayShapleyValue(TestMetric):
    METRIC = AtMost1DegreeAwayShapleyValue

    TEST_CASES = (
        (get_4_free_nodes, [1, 1, 1, 1]),
        (get_4_elements_list_graph, [5.0/6, 1.166, 1.166, 5.0/6]),
        (get_4_elements_clique, [1, 1, 1, 1]),
        (get_11_elements_star, [
            5.090, 0.590, 0.590, 0.590, 0.590, 0.590,
            0.590, 0.590, 0.590, 0.590, 0.590
        ]),
    )


class TestsAtMost2DegreeAwayShapleyValue(TestMetric):
    METRIC = partial(AtMostKDegreeAwayShapleyValue, infection_factor=2)

    # TODO Check values
    TEST_CASES = (
        (get_4_free_nodes, [1, 1, 1, 1]),
        (get_4_elements_list_graph, [1.166, 1.166, 1.166, 1.166]),
        (get_4_elements_clique, [1.5, 1.5, 1.5, 1.5]),
        (get_11_elements_star, [
            1, 1.081, 1.081, 1.081, 1.081, 1.081,
            1.081, 1.081, 1.081, 1.081, 1.081
        ]),
    )


if __name__ == '__main__':
    TestsDegreeMetric()
    TestsBetweennessMetric()
    TestsClosenessMetric()
    TestsEigenVectorMetric()
    TestsSecondOrderDegreeMassMetric()
    TestsAtMost1DegreeAwayShapleyValue()
    TestsAtMost2DegreeAwayShapleyValue()
