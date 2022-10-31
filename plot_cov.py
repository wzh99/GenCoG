import numpy as np
from matplotlib import pyplot as plt

names = ['LEMON', 'Muffin-Chain', 'Muffin-Cell', 'Luo-WS', 'Luo-RN', 'GenCoG-M', 'GenCoG']
paths = [
    'out/cov-lemon/data.txt',
    'out/cov-graphfuzz-ws/data.txt',
    'out/cov-graphfuzz-rn/data.txt',
    'out/cov-muffin-dag/data.txt',
    'out/cov-muffin-template/data.txt',
    'out/cov-gencog-muffin/data.txt',
    'out/cov-gencog-complete/data.txt',
]
colors = ['deeppink', 'tab:purple', 'dodgerblue', 'tab:cyan', 'tab:green', 'darkorange', 'crimson']
assert len(names) == len(paths), len(names) == len(colors)

data = [np.loadtxt(p) for p in paths]

plt.rc('font', family='Latin Modern Sans', size=9)

plt.figure(figsize=(5, 3.5))
plt.gca().set_box_aspect(3 / 5)
for i in range(len(names)):
    plt.plot(data[i][:, 0], data[i][:, 1], color=colors[i], linewidth=2, label=names[i])
plt.ylim(3.05e4, 3.5e4)
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xlabel('#Vertices')
plt.ylabel('Line Coverage')
plt.legend(ncol=3, loc='lower right')
plt.savefig('out/cov.pdf')
plt.show()
