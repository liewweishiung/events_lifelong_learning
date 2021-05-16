"""
This file is copied from
https://github.com/GMvandeVen/brain-inspired-replay/tree/master/visual

Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).
"""

import matplotlib
matplotlib.use('Agg')
# above 2 lines set the matplotlib backend to 'Agg', which
#  enables matplotlib-plots to also be generated if no X-server
#  is defined (e.g., when running in basic Docker-container)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn as sns

plt.style.use('seaborn')

tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)


def open_pdf(full_path):
    return PdfPages(full_path)


def plot_scatter(x, y, colors=None, ylabel=None, xlabel=None, title=None, top_title=None, names=None,
                 xlim=None, ylim=None, markers=None, file_path=None):
    '''Generate a figure containing a scatter-plot.'''

    # if needed, generate default point-names
    if names == None:
        n_points = len(y)
        names = ["point " + str(id) for id in range(n_points)]

    # make plot
    f, axarr = plt.subplots(1, 1, figsize=(12, 7))

    # finish layout
    # -set y/x-axis
    if ylim is not None:
        axarr.set_ylim(ylim)
    if xlim is not None:
        axarr.set_xlim(xlim)
    # -add axis-labels
    if xlabel is not None:
        axarr.set_xlabel(xlabel)
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    # -add title(s)
    if title is not None:
        axarr.set_title(title)
    if top_title is not None:
        f.suptitle(top_title)
    # -add legend
    if names is not None:
        axarr.legend()

    g = sns.scatterplot(
        ax=axarr,
        x=x,
        y=y,
        hue=colors,
        palette=sns.color_palette("bright", len(np.unique(colors))),
        legend="full",
        style=colors,
        alpha=0.3
    )

    # return the figure
    g.figure.savefig("{}/tsne_classes-{}.pdf".format(file_path, len(np.unique(colors))),
                     format="pdf",
                     bbox_inches="tight")



def plot_lines(list_with_lines, x_axes=None, line_names=None, colors=None, title=None,
               title_top=None, xlabel=None, ylabel=None, ylim=None, figsize=None, list_with_errors=None, errors="shaded",
               x_log=False, with_dots=True, linestyle='solid', h_line=None, h_label=None, h_error=None,
               h_lines=None, h_colors=None, h_labels=None, h_errors=None, markers=None, font_scale=1.0):
    '''Generates a figure containing multiple lines in one plot.

    :param list_with_lines: <list> of all lines to plot (with each line being a <list> as well)
    :param x_axes:          <list> containing the values for the x-axis
    :param line_names:      <list> containing the names of each line
    :param colors:          <list> containing the colors of each line
    :param title:           <str> title of plot
    :param title_top:       <str> text to appear on top of the title
    :return: f:             <figure>
    '''

    # if needed, generate default x-axis
    if x_axes == None:
        n_obs = len(list_with_lines[0])
        x_axes = list(range(n_obs))

    # if needed, generate default line-names
    if line_names == None:
        n_lines = len(list_with_lines)
        line_names = ["line " + str(line_id) for line_id in range(n_lines)]

    # make plot
    size = (12,7) if figsize is None else figsize
    f, axarr = plt.subplots(1, 1, figsize=size)
    axarr.xaxis.set_ticks(np.arange(int(max(x_axes) / len(x_axes)),
                                    int(max(x_axes) + 1),
                                    int(max(x_axes) / len(x_axes)) #* (5 if len(x_axes) > 16 else 2)
                                    )
                          )
    axarr.set_xticklabels(np.arange(int(max(x_axes) / len(x_axes)),
                                    int(max(x_axes) + 1),
                                    int(max(x_axes) / len(x_axes)) #* (5 if len(x_axes) > 16 else 2)
                                    ),
                          rotation=0,#45 if max(x_axes) > 12 else 0,
                          fontsize=16*font_scale)

    for tick in axarr.yaxis.get_major_ticks():
        tick.label.set_fontsize(16*font_scale)

        # add error-lines / shaded areas
    if list_with_errors is not None:
        for task_id, name in enumerate(line_names):
            if errors == "shaded":
                axarr.fill_between(x_axes, list(np.array(list_with_lines[task_id]) + np.array(list_with_errors[task_id])),
                                   list(np.array(list_with_lines[task_id]) - np.array(list_with_errors[task_id])),
                                   color=None if (colors is None) else colors[task_id], alpha=0.25)
            else:
                axarr.plot(x_axes, list(np.array(list_with_lines[task_id]) + np.array(list_with_errors[task_id])), label=None,
                           color=None if (colors is None) else colors[task_id], linewidth=1, linestyle='dashed')
                axarr.plot(x_axes, list(np.array(list_with_lines[task_id]) - np.array(list_with_errors[task_id])), label=None,
                           color=None if (colors is None) else colors[task_id], linewidth=1, linestyle='dashed')

    # mean lines
    for task_id, name in enumerate(line_names):
        axarr.plot(x_axes, list_with_lines[task_id], label=name,
                   color=colors[task_id],
                   linewidth=1, marker=markers[task_id] if with_dots else None, linestyle=linestyle if type(linestyle)==str else linestyle[task_id])

    # Chance line:
    values = np.ones((len(x_axes),)) / np.arange(x_axes[0], x_axes[-1] + x_axes[0], x_axes[0])
    axarr.plot(x_axes, values, label="Chance", color="grey")

    # add horizontal line
    if h_line is not None:
        axarr.axhline(y=h_line, label=h_label, color="grey")
        if h_error is not None:
            if errors == "shaded":
                axarr.fill_between([x_axes[0], x_axes[-1]],
                                   [h_line + h_error, h_line + h_error], [h_line - h_error, h_line - h_error],
                                   color="grey", alpha=0.25)
            else:
                axarr.axhline(y=h_line + h_error, label=None, color="grey", linewidth=1, linestyle='dashed')
                axarr.axhline(y=h_line - h_error, label=None, color="grey", linewidth=1, linestyle='dashed')

    # add horizontal lines
    if h_lines is not None:
        h_colors = colors if h_colors is None else h_colors
        for task_id, new_h_line in enumerate(h_lines):
            axarr.axhline(y=new_h_line, label=None if h_labels is None else h_labels[task_id],
                          color=None if (h_colors is None) else h_colors[task_id])
            if h_errors is not None:
                if errors == "shaded":
                    axarr.fill_between([x_axes[0], x_axes[-1]],
                                       [new_h_line + h_errors[task_id], new_h_line+h_errors[task_id]],
                                       [new_h_line - h_errors[task_id], new_h_line - h_errors[task_id]],
                                       color=None if (h_colors is None) else h_colors[task_id], alpha=0.25)
                else:
                    axarr.axhline(y=new_h_line+h_errors[task_id], label=None,
                                  color=None if (h_colors is None) else h_colors[task_id], linewidth=1,
                                  linestyle='dashed')
                    axarr.axhline(y=new_h_line-h_errors[task_id], label=None,
                                  color=None if (h_colors is None) else h_colors[task_id], linewidth=1,
                                  linestyle='dashed')

    # finish layout
    # -set y-axis
    if ylim is not None:
        axarr.set_ylim(ylim)
    # -add axis-labels
    if xlabel is not None:
        axarr.set_xlabel(xlabel, fontsize=22*font_scale)
    if ylabel is not None:
        axarr.set_ylabel(ylabel, fontsize=22*font_scale)
    # -add title(s)
    if title is not None:
        axarr.set_title(title, fontsize=22*font_scale)
    if title_top is not None:
        f.suptitle(title_top)
    # -add legend
    if line_names is not None:
        axarr.legend(fontsize=15*font_scale, ncol=2)
    # -set x-axis to log-scale
    if x_log:
        axarr.set_xscale('log')

    # return the figure
    return f















