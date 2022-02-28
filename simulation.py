import numpy as np
import collections
from itertools import chain, repeat, count, cycle, islice, product
import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

sns.set_style("dark")


class RLModel:
    def __init__(self, f, eta_h, eta_v, suffix):
        self.f = f
        self.eta_h = eta_h
        self.eta_v = eta_v
        self.delta = 0
        self.h = 0
        self.m = 0
        self.v = 0
        self.suffix = suffix
        self.history = collections.defaultdict(list)

    def step(self, reward):
        # update perceived reward and prediction error
        r_perceived = reward * (self.f ** self.m)
        self.delta = r_perceived - self.v

        # update mood
        self.h = self.h + self.eta_h * (self.delta - self.h)
        self.m = np.tanh(self.h)

        # update expected value
        self.v = self.v + self.eta_v * self.delta

        # append to history
        self.history["expected value"].append(self.v)
        self.history["mood"].append(self.m)
        self.history["perceived reward"].append(r_perceived)

    def simulate(self, iters, reward):
        # run the simulation for a given number of iterations
        for _ in range(iters):
            self.step(reward)
        return self.history

    def plot(self):
        utils.save_lineplot(
            y=self.history["expected value"],
            ylim=(0, 20),
            title=f"F-value = {self.f}",
            ylabel="Expected Value",
            xlabel="Iteration",
            filename=f"img/expected_value_{self.suffix}.png",
            hline=10,
        )
        utils.save_lineplot(
            y=self.history["mood"],
            ylim=(-1.5, 1.5),
            title=f"F-value = {self.f}",
            ylabel="Mood",
            xlabel="Iteration",
            filename=f"img/mood_{self.suffix}.png",
        )
        utils.save_lineplot(
            y=self.history["perceived reward"],
            ylim=(0, 20),
            title=f"F-value = {self.f}",
            ylabel="Perceived Reward",
            xlabel="Iteration",
            filename=f"img/perceived_reward_{self.suffix}.png",
            hline=10,
        )


def choice_experiment():
    high_HPS_win_WOF = RLModel(f=1.6, eta_h=0.28, eta_v=0.15, suffix="high_HPS_win_WOF")
    low_HPS_win_WOF = RLModel(f=0.85, eta_h=0.28, eta_v=0.15, suffix="low_HPS_win_WOF")
    high_HPS_lose_WOF = RLModel(
        f=1.6, eta_h=0.28, eta_v=0.15, suffix="high_HPS_lose_WOF"
    )
    low_HPS_lose_WOF = RLModel(
        f=0.85, eta_h=0.28, eta_v=0.15, suffix="low_HPS_lose_WOF"
    )

    # rewards_win_WOF = chain(repeat(10, 10), repeat(20, 5), repeat(10, 10))
    # rewards_lose_WOF = chain(repeat(10, 10), repeat(0, 5), repeat(10, 10))

    rewards_win_WOF = islice(cycle(range(10)), 100)
    rewards_lose_WOF = islice(cycle(list(range(10))[::-1]), 100)

    for r in rewards_win_WOF:
        high_HPS_win_WOF.step(r)
        low_HPS_win_WOF.step(r)

    for r in rewards_lose_WOF:
        high_HPS_lose_WOF.step(r)
        low_HPS_lose_WOF.step(r)

    models = [high_HPS_win_WOF, low_HPS_win_WOF, high_HPS_lose_WOF, low_HPS_lose_WOF]

    dfs = [
        pd.DataFrame(
            {
                "Condition": m.suffix,
                "Iteration": range(len(m.history["expected value"])),
                "Value": m.history["expected value"],
            }
        )
        for m in models
    ]
    df = pd.concat(dfs)

    plot = sns.lineplot(data=df, x="Iteration", y="Value", hue="Condition")
    plt.savefig("img/choice_experiment.png", dpi=300)


def eta_sweep():
    eta_hs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    eta_vs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

    data = []

    for eta_h, eta_v in product(eta_hs, eta_vs):
        model = RLModel(f=1.6, eta_h=eta_h, eta_v=eta_v, suffix=f"{eta_h}_{eta_v}")
        model.simulate(300, reward=10)
        moods = model.history["mood"]
        peak_idxs, _ = find_peaks(moods, distance=5)
        if len(peak_idxs) != 0:
            period = len(moods) / len(peak_idxs)
        else:
            period = np.nan
        freq = 1.0 / period
        data.append({"eta_h": eta_h, "eta_v": eta_v, "period": period, "freq": freq})

    df = pd.DataFrame(data)
    df = df.pivot(index="eta_h", columns="eta_v", values="freq")

    sns.heatmap(data=df, cmap="viridis")
    plt.savefig("img/eta_sweep.png", dpi=300)


if __name__ == "__main__":

    # fs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    # suffixes = [str(x).replace(".", "-") for x in fs]
    # histories = []
    # for f, suffix in zip(fs, suffixes):
    #     model = RLModel(f=f, eta_h=0.1, eta_v=0.1, suffix=suffix)
    #     histories.append(model.simulate(iters=500, reward=10))
    #     # model.plot()
    # utils.save_parameter_lineplot(histories, fs)

    # eta_hs = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    # suffixes = [str(x).replace(".", "-") for x in eta_hs]
    # histories = []
    # for eta, suffix in zip(eta_hs, suffixes):
    #     model = RLModel(f=1.6, eta_h=eta, eta_v=0.1, suffix=suffix)
    #     histories.append(model.simulate(iters=500, reward=10))
    #     # model.plot()
    # utils.save_parameter_lineplot(histories, eta_hs)

    # choice_experiment()

    eta_sweep()
