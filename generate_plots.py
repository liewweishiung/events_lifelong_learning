"""
Generation of figures
2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
y = np.random.rand(100) * 0.2
y[10] = 0.8
y[20] = 0.5
y[30] = 1
y[50] = 0.6
y[60] = 0.4
y[70] = 0.4
y[80] = 0.6
x = np.arange(0, 100)
sns.barplot(x=x, y=y)
plt.show()


import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

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


def set_size(width=345, fraction=1):
    # This method is taken from https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


sns.set(rc={'figure.figsize': (set_size())})

hE = torch.load('./store/hE.pt').detach().cpu()
habituation_counter = torch.ones(2000)
idx_neurons = torch.topk(hE, math.floor(0.02 * 2000), dim=1).indices
hE = hE.numpy()
idx_neurons = idx_neurons.numpy()

clrs = ['red' if x in idx_neurons[32, :] else 'grey' for x in np.arange(0, 2000)]
# sns.set(font_scale=1.2)
g = sns.barplot(x=np.arange(1, 2001), y=hE[32, :], palette=clrs)

x_indices = [i * 200 - 1 for i in range(1, 11)]
x_indices.insert(0, 0)
g.set(xticks=x_indices)
g.set_ylabel("Activation value")
g.set_xlabel("Index of a neuron")
g.set_title("Activation values of neurons")
x_indices = [i + 1 for i in x_indices]
g.set_xticklabels(labels=x_indices, rotation=45)
g.figure.savefig("./store/plots/activations.pdf", format="pdf", bbox_inches="tight")
plt.close()

sns.set(font_scale=1.4)
h = np.load('./store/results/habs_over_episodes_0.001.npy')
d = {'Episode': range(1, len(h) + 1), 'Percentage': h * 100}
df = pd.DataFrame(data=d)
g = sns.barplot(x="Episode", y="Percentage", data=df, palette="Blues_d")
g.set_xlabel("Episode")
g.set_ylabel("Percentage")
g.set_title("Percentage of habituation counters \n larger than 0.01")
g.set_xticklabels(labels=range(1, len(h) + 1), rotation=45)
g.figure.savefig("./store/plots/habs_over_episodes_0.001.pdf", format="pdf", bbox_inches="tight")
plt.close()

sns.set(font_scale=1.4)
h = torch.load('./store/habituation_counter_setup_3.pt').cpu().numpy()
g = sns.histplot(x=h, bins=10)
g.set_xlabel("Habituation counter")
g.set_title("Distribution of habituation counters")
g.figure.savefig("./store/plots/distribution_habituation_long_horizon_bir_setup3.pdf", format="pdf",
                 bbox_inches="tight")
plt.close()

sns.set(font_scale=1.4)
h = torch.load('./store/habituation_counter_setup1.pt').cpu().numpy()
g = sns.histplot(x=h, bins=10)
g.set_xlabel("Habituation counter")
g.set_title("Distribution of habituation counters")
g.figure.savefig("./store/plots/distribution_habituation_short_horizon_bir_setup1.pdf", format="pdf",
                 bbox_inches="tight")
plt.close()

sns.set(font_scale=1.4)
h = torch.load('./store/habituation_counter_setup2.pt').cpu().numpy()
g = sns.histplot(x=h, bins=10)
g.set_xlabel("Habituation counter")
g.set_title("Distribution of habituation counters")
g.figure.savefig("./store/plots/distribution_habituation_short_horizon_bir_setup2.pdf", format="pdf",
                 bbox_inches="tight")
plt.close()


def habituate(habituations, decay_rate=0.2):
    for i in range(0, len(habituations)):
        if i == 0:
            habituations[i] = 1
        elif i == 1:
            habituations[i] = 1 + decay_rate * (1 - 1) - decay_rate
        else:
            habituations[i] = habituations[i - 1] + decay_rate * (1 - habituations[i - 1]) - decay_rate


x_values = np.arange(0, 101)
habituations_1 = np.zeros(101)
habituate(habituations_1, decay_rate=0.001)

habituations_2 = np.zeros(101)
habituate(habituations_2, decay_rate=0.2)

d = {'Iteration': x_values, '0.001': habituations_1, '0.2': habituations_2}
df = pd.DataFrame(data=d)

df = df.melt('Iteration', value_vars=['0.001', '0.2'], value_name='Value')
df.rename(columns={"variable": "Decay rate"}, inplace=True)
sns.set(font_scale=1.2)
g = sns.lineplot(x="Iteration", y="Value", hue='Decay rate', data=df, style='Decay rate')
x_indices = [i * 20 for i in range(1, 6)]
x_indices.insert(1, 1)
g.set(xticks=x_indices)
g.set_title("Habituation counter")
g.figure.savefig("./store/plots/habituation_counter.pdf",
                 format="pdf",
                 bbox_inches="tight")
plt.close()
