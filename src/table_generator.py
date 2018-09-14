import sys


TABLE_PREFIX = (
r"""\begin{table}[!ht]
\centering
\begin{tabular}{| c | c | c | c | c | c | c |}
\rot{}
& \rot{ROAM(1)}
& \rot{ROAM(2)}
& \rot{ROAM(3)}
& \rot{ROAM(4)}
& \rot{AVERAGE}
\\
\hline
""")

PATTERN = r"{} & {} & {} & {} & {} & {}\\"


TABLE_SUFIX = (
r"""
\hline
\end{tabular}
\caption{$absolute\ scores\ table$}
\end{table}
""")


with open(sys.argv[1]) as file_:
    lines = file_.read().split('\n')
    tmp = []
    for i in range(1, 7):
        tmp.append(lines[i * 18: (i + 1) * 18])

    output1 = [PATTERN.format(*tup) for tup in map(tuple, zip(*tmp))]
    output2 = [s.replace('_', ' ') for s in output1[1:]]

    print(
        TABLE_PREFIX +
        '\n\\hline\n'.join(output2) +
        TABLE_SUFIX
    )


# degree & 2 & 9 & 4 & 4 & 1 & 1 & 3.5\\
