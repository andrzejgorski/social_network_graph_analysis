from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import os
import pickle


def save_metric_ranking_plot(results, metric_name, label, output_file=None):
    output_format = '.jpeg'

    fig, ax = plt.subplots()
    plt.title(metric_name)
    colors = ('purple', 'green', 'r', 'c', 'm', 'y', 'k', 'w')
    shapes = ('s', '^', 'o', 'v', 'D', 'p', 'x', '8')
    linestyles = ((0, (15, 10, 3, 10)), '--', ':', '-.')

    for i in range(len(results)):
        label_index = label + str(i + 1)
        line = plt.plot(
            list(map(lambda x: x + 1, results[i])), label=label_index)
        plt.setp(
            line, marker=shapes[i], markersize=15.0, markeredgewidth=2,
            markerfacecolor="None", markeredgecolor=colors[i],
            linewidth=2, linestyle=linestyles[i], color=colors[i]
        )

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().invert_yaxis()
    plt.legend(loc='lower left')
    plt.margins(0.1)
    plt.xlabel("iterations")
    plt.ylabel("ranking")

    output_file = output_file or metric_name + output_format

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def save_metric_ranking_plot_for_random_graphs(results, metric_name, label,
                                               output_file=None):
    output_format = '.jpeg'

    plt.subplots()
    plt.title(metric_name)
    colors = ('purple', 'green', 'r', 'c', 'm', 'y', 'k', 'w')
    shapes = ('s', '^', 'o', 'v', 'D', 'p', 'x', '8')
    linestyles = ((0, (15, 10, 3, 10)), '--', ':', '-.')

    for i in range(len(results)):
        label_index = label + str(i + 1)
        line = plt.plot(
            list(map(lambda x: x[0] + 1, results[i])), label=label_index
        )
        plt.setp(
            line, marker=shapes[i], markersize=15.0, markeredgewidth=2,
            markerfacecolor="None", markeredgecolor=colors[i], linewidth=2,
            linestyle=linestyles[i], color=colors[i]
        )
        plt.fill_between(range(len(results[i])),
                         list(map(lambda x: x[1][0] + 1, results[i])),
                         list(map(lambda x: x[1][1] + 1, results[i])),
                         facecolor=colors[i], edgecolors=None,
                         alpha=0.2)

    plt.gca().invert_yaxis()
    plt.legend(loc='lower left')
    plt.margins(0.1)
    plt.xlabel("iterations")
    plt.ylabel("ranking")

    output_file = output_file or metric_name + output_format

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def save_influence_value_plot(metric_values, influence_name, label, dir_name):
    output_format = '.pdf'

    plt.subplots()
    plt.title(influence_name)
    colors = ('purple', 'green', 'r', 'c', 'm', 'y', 'k', 'w')
    shapes = ('s', '^', 'o', 'v', 'D', 'p', 'x', '8')
    linestyles = ((0, (15, 10, 3, 10)), '--', ':', '-.')

    for i in range(len(metric_values)):
        label_index = label + str(i + 1)
        line = plt.plot(
            list(map(lambda x: x / metric_values[i][0], metric_values[i])), label=label_index
        )
        plt.setp(
            line, marker=shapes[i], markersize=15.0, markeredgewidth=2,
            markerfacecolor="None", markeredgecolor=colors[i], linewidth=2,
            linestyle=linestyles[i], color=colors[i]
        )

    plt.legend(loc='lower left')
    plt.margins(0.1)
    plt.xlabel("iterations")
    plt.ylabel("value")
    output_file = os.path.join(dir_name, influence_name + output_format)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def save_influence_value_plot_for_random_graphs(metric_values, influence_name, label, dir_name):
    output_format = '.pdf'

    plt.subplots()
    plt.title(influence_name)
    colors = ('purple', 'green', 'r', 'c', 'm', 'y', 'k', 'w')
    shapes = ('s', '^', 'o', 'v', 'D', 'p', 'x', '8')
    linestyles = ((0, (15, 10, 3, 10)), '--', ':', '-.')

    for i in range(len(metric_values)):
        label_index = label + str(i + 1)
        line = plt.plot(
            list(map(lambda x: x[0] / metric_values[i][0][0], metric_values[i])), label=label_index
        )
        plt.setp(
            line, marker=shapes[i], markersize=15.0, markeredgewidth=2,
            markerfacecolor="None", markeredgecolor=colors[i], linewidth=2,
            linestyle=linestyles[i], color=colors[i]
        )
        plt.fill_between(range(len(metric_values[i])),
                         list(map(lambda x: x[1][0] / metric_values[i][0][0], metric_values[i])),
                         list(map(lambda x: x[1][1] / metric_values[i][0][0], metric_values[i])),
                         facecolor=colors[i], edgecolors=None,
                         alpha=0.2)

    plt.legend(loc='lower left')
    plt.margins(0.1)
    plt.xlabel("iterations")
    plt.ylabel("value")
    output_file = os.path.join(dir_name, influence_name + output_format)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def save_scores_table(scores_table, label, output_file='scores_table.pdf'):
    sorted_scores = sorted(scores_table, key=lambda score: score[len(score) - 1])

    sorted_scores = [
        [str(round(x, 3)) if type(x) != str else x for x in y]
        for y in sorted_scores
    ]

    for i in range(len(sorted_scores)):
        sorted_scores[i].insert(0, str(i+1))

    plt.figure()
    fig, ax = plt.subplots()

    column_labels = ['Pos.'] +\
                    ['Metric name'] +\
                    [label + '(' + str(i + 1) + ')' for i in range(len(sorted_scores[0]) - 3)] +\
                    ['AVERAGE']

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(
        cellText=sorted_scores,
        colLabels=column_labels,
        colWidths=[0.1] + [0.4] + [0.1] * 5, loc='upper center'
    )
    fig.savefig(output_file)
    plt.close()

    with open(output_file + '.p', 'wb') as f:
        pickle.dump(sorted_scores, f)
