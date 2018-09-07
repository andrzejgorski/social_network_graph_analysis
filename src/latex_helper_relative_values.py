#!/usr/bin/python

import pickle
import sys

accumulated = []

print("\\begin{table}[!ht]")
print("\\centering")
print("\\begin{tabular}{|", end='')
for i in range(len(sys.argv) + 1):
    print(' c |', end='')
print("}")

print('\\rot{}')

for results in sys.argv[1:]:
    with open(results, 'rb') as f:
        accumulated += [(results, pickle.load(f))]

for name, results in accumulated:
    print(' & \\rot{' + name.split('/')[0].replace("_", " ") + '}')

print(' & \\rot{AVERAGE}')

print('\\\\\n\\hline')

metrics = []
name, results = accumulated[0]
for metric in results:
    metrics += [metric[1]]

metric_positions = {k: [] for k in metrics}

for name, results in accumulated:
    min_result = min(results, key=lambda y: float(y[-1]))
    min_result = float(min_result[-1]) + 0.01
    for m in results:
        m.append((float(m[-1]) + 0.01) / float(min_result))

for metric in metrics:
    for name, results in accumulated:
        for m in results:
            if m[1] == metric:
                metric_positions[metric].append(m[-1])

for result in metric_positions.values():
    result.append(sum(result) / len(result))

list_of_results = []

for k, v in metric_positions.items():
    list_of_results.append([k] + v)

list_of_results.sort(key=lambda y: y[len(y) - 1])

for metric_results in list_of_results:
    print(metric_results[0].replace("_", " "), end='')
    for x in metric_results[1:]:
        print(' & ' + str(round(x, 2)), end='')
    print('\\\\\n\\hline')

print("\\end{tabular}")
print("\\end{table}")
