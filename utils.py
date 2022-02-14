import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")


def save_lineplot(
    y,
    title="Plot",
    xlabel="X Axis",
    ylabel="Y Axis",
    filename="plot.png",
    ylim=None,
    hline=None,
    titlefontsize=20,
    xyfontsize=18,
    dpi=300,
):
    plt.clf()
    x = range(len(y))
    plot = sns.lineplot(x=x, y=y)
    plot.set_ylim(ylim)
    plot.set_title(title, fontsize=titlefontsize)
    plot.set_xlabel(xlabel, fontsize=xyfontsize)
    plot.set_ylabel(ylabel, fontsize=xyfontsize)
    if hline != None:
        plot.axhline(hline, linestyle="--")
    plt.savefig(filename, dpi=dpi)
