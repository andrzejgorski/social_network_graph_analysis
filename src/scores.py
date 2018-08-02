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
