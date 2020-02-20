import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.signal import savgol_filter


def plot_line_graph(xvals, list_of_yvals, list_of_names,
                    filename, title, log_scale=False):
    plt.clf()
    max_x = 0
    for i in range(len(list_of_yvals)):
        plt.plot(xvals[i], savgol_filter(list_of_yvals[i], 7, 3), label=list_of_names[i])
        max_x = max(max(xvals[i]), max_x)

    # Create legend & Show graphic
    plt.legend()

    if log_scale:
        plt.yscale('log')
    else:
        plt.yscale('linear')

    t = [0, max_x]
    plt.xticks(t, t)

    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
