def calculate_integral_score(results):
    score = 0
    for i in range(len(results) - 1):
        score += (results[i] + results[i + 1]) / 2.0
    return score


def calculate_relative_integral_score(results):
    score = 0
    integral_base = results[0]
    for i in range(len(results) - 1):
        score += (results[i] + results[i + 1] - 2 * integral_base) / 2.0
    return score


def get_ranking_scores(ranking_results, metric_name=None):
    scores = []
    shifted_scores = []

    for result in ranking_results:
        scores.append(calculate_integral_score(result))
        shifted_scores.append(calculate_relative_integral_score(result))

    scores.append(sum(scores) / float(len(scores)))
    if metric_name:
        scores.insert(0, metric_name)
    shifted_scores.append(sum(shifted_scores) / float(len(shifted_scores)))
    if metric_name:
        shifted_scores.insert(0, metric_name)

    return scores, shifted_scores
