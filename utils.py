import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("dark")


def save_lineplot(
    y,
    title="Plot",
    xlabel="X Axis",
    ylabel="Y Axis",
    filename="plot.png",
    ylim=None,
    hline=None,
    titlefontsize=18,
    xyfontsize=16,
    dpi=300,
):
    """Convenience function to create a line plot and save it to a specified file in a single line of code.

    Args:
        y (List[float]): list of values to plot
        title (str, optional): _description_. Defaults to "Plot".
        xlabel (str, optional): _description_. Defaults to "X Axis".
        ylabel (str, optional): _description_. Defaults to "Y Axis".
        filename (str, optional): _description_. Defaults to "plot.png".
        ylim (Tuple[float, float], optional): _description_. Defaults to None.
        hline (float, optional): y value at which to draw a horizontal dashed line. Defaults to None.
        titlefontsize (int, optional): _description_. Defaults to 20.
        xyfontsize (int, optional): _description_. Defaults to 18.
        dpi (int, optional): dots per square inch for saving. Defaults to 300.
    """
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


def save_parameter_lineplot(histories, fs):
    plt.clf()
    ax = plt.gca()
    save_multiple_lineplot(
        ys=[h['perceived reward'] for h in histories],
        params=fs,
        ylim=(0, 20),
        # title=f"F-value = {self.f}",
        ylabel="Perceived Reward",
        xlabel="Iteration",
        filename="img/perceived_reward_sweep.png",
        hline=10,
        ax=ax,
    )
    save_multiple_lineplot(
        ys=[h['expected value'] for h in histories],
        params=fs,
        ylim=(0, 20),
        # title=f"F-value = {self.f}",
        ylabel="Expected Value",
        xlabel="Iteration",
        filename="img/expected_value_sweep.png",
        hline=10,
        ax=ax,
    )
    save_multiple_lineplot(
        ys=[h['mood'] for h in histories],
        params=fs,
        ylim=(-1.5, 1.5),
        # title=f"F-value = {self.f}",
        ylabel="Expected Value",
        xlabel="Iteration",
        filename="img/mood_sweep.png",
        hline=10,
        ax=ax,
    )

def save_multiple_lineplot(
        ys, 
        params, 
        title="Plot",
        xlabel="X Axis",
        ylabel="Y Axis",
        filename="plot.png",
        ylim=None,
        hline=None,
        titlefontsize=18,
        xyfontsize=16,
        dpi=300,
        ax=plt.gca(),
    ):

    col = plt.cm.cool(np.linspace(start=0, stop=1, num=len(ys)))    
    for (y, p, c) in zip(ys, params, col):
        x = range(len(y))
        plot = sns.lineplot(x=x, y=y, color=c, ax=ax)
        plot.set_ylim(ylim)
        plot.set_title(title, fontsize=titlefontsize)
        plot.set_xlabel(xlabel, fontsize=xyfontsize)
        plot.set_ylabel(ylabel, fontsize=xyfontsize)
        if hline != None:
            plot.axhline(hline, linestyle="--")
    plt.savefig(filename, dpi=dpi)
    