import numpy as np
from matplotlib import pyplot as plt

names = ['Muffin-Chain', 'Muffin-Cell', 'DeepTyper-M']
paths = [
    'out/muffin-dag.txt',
    'out/muffin-template.txt',
    'out/typefuzz.txt'
]
colors = ['deepskyblue', 'limegreen', 'darkorange']
assert len(names) == len(paths), len(names) == len(colors)

data = [np.loadtxt(p) for p in paths]

plt.rc('font', family='Latin Modern Sans', size=14)

plt.figure(figsize=(6, 4))
plt.gca().set_box_aspect(3 / 5)
for i in range(len(names)):
    plt.plot(data[i][:, 0], data[i][:, 1], color=colors[i], linewidth=2, label=names[i])
plt.ylim(0, 0.6)
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.xlabel('#Vertices')
plt.ylabel('Vertex Diversity')
plt.legend()
plt.savefig('out/vert-div.pdf')
plt.show()

plt.figure(figsize=(6, 4))
plt.gca().set_box_aspect(3 / 5)
for i in range(len(names)):
    plt.plot(data[i][:, 0], data[i][:, 2], color=colors[i], linewidth=2, label=names[i])
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.ylim(0, 0.85)
plt.xlabel('#Vertices')
plt.ylabel('Edge Diversity')
plt.legend()
plt.savefig('out/edge-div.pdf')
plt.show()