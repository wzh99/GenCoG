import numpy as np
from matplotlib import pyplot as plt

names = ['Muffin-Chain', 'Muffin-Cell', 'DeepTyper-M', 'DeepTyper']
paths = [
    'out/cov-muffin-dag/data.txt',
    'out/cov-muffin-template/data.txt',
    'out/cov-typefuzz-muffin/data.txt',
    'out/cov-typefuzz-complete/data.txt',
]
colors = ['deepskyblue', 'limegreen', 'darkorange', 'crimson']
assert len(names) == len(paths), len(names) == len(colors)

data = [np.loadtxt(p) for p in paths]

plt.rc('font', family='Latin Modern Sans', size=10)

plt.figure(figsize=(5, 3))
plt.gca().set_box_aspect(1 / 2)
for i in range(len(names)):
    plt.plot(data[i][:, 0], data[i][:, 1], color=colors[i], linewidth=2, label=names[i])
plt.ylim(3.1e4, 3.5e4)
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xlabel('#Vertices')
plt.ylabel('Line Coverage')
plt.legend()
plt.savefig('out/cov.pdf')
plt.show()
