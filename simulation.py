import numpy as np
import collections
import utils


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



if __name__ == "__main__":

    fs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    suffixes = [str(x).replace(".", "-") for x in fs]
    histories = []
    for f, suffix in zip(fs, suffixes):
        model = RLModel(f=f, eta_h=0.1, eta_v=0.1, suffix=suffix)
        histories.append(model.simulate(iters=500, reward=10))
        # model.plot()
    utils.save_parameter_lineplot(histories, fs)



