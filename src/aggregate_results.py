import sys
from pprint import pprint

pos_results = dict()
inf_results = dict()

def prepare_results(table):
    table = map(lambda x: (x[0], sum(x[1]) / len(x[1])), table.items())
    table = sorted(table, key=lambda x: x[1], reverse=True)
    table = zip(table, range(1, len(table) + 1))
    table = dict(map(lambda x: (x[0][0], (round(x[0][1], 3), x[1])), table))
    return table

def end():
    pos_aggr = prepare_results(pos_results)
    inf_aggr = prepare_results(inf_results)

    final_results = dict()
    for k in pos_aggr.keys():
        final_results[k] = (pos_aggr[k] + inf_aggr[k],
                            pos_aggr[k][1] + inf_aggr[k][1])

    final_results = sorted(final_results.items(), key=lambda x: x[1][1])
    pprint(final_results)
    exit()

with open(sys.argv[1]) as f:
    strategy = f.readline()
    while True:
        if not strategy in pos_results:
            pos_results[strategy] = []
            inf_results[strategy] = []
        result = f.readline()
        while True:
            if not result.startswith("INFLUENCE"):
                result = float(result)
                pos_results[strategy].append(result)
            else:
                result = f.readline()
                result = float(result)
                inf_results[strategy].append(result)
                result = f.readline()  # eat INFLUENCE string
                result = f.readline()
                result = float(result)
                inf_results[strategy].append(result)
            result = f.readline()
            if not result:
                end()
            try:
                float(result)
            except Exception:
                if not result.startswith("INFLUENCE"):
                    strategy = result
                    break
