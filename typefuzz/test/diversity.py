from argparse import Namespace, ArgumentParser

import numpy as np
from numpy.random import Generator, PCG64
from tqdm import trange

from typefuzz.config import muffin_ops
from typefuzz.graph import GraphGenerator
from typefuzz.metric.div import EdgeDiversity, VertexDiversity
from typefuzz.spec import OpRegistry, TypeSpec

options = Namespace()


def parse_args():
    global options
    p = ArgumentParser()
    p.add_argument('-n', '--number', type=int, help='Number of graphs to generate.')
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    options = p.parse_args()


def main():
    TypeSpec.for_graph = True
    rng = Generator(PCG64(seed=options.seed))
    ops = [OpRegistry.get(name) for name in muffin_ops]
    gen = GraphGenerator(ops, rng)
    vert_div = VertexDiversity(ops)
    edge_div = EdgeDiversity(ops)
    for _ in trange(options.number):
        graph = gen.generate()
        vert_div.evaluate(graph)
        edge_div.evaluate(graph)
    np.set_printoptions(precision=3)
    print(type(vert_div).__name__, vert_div.result)
    print(type(edge_div).__name__, edge_div.result)


if __name__ == '__main__':
    parse_args()
    main()

"""
penalty=4
[0.197 0.183 0.194 0.194 0.191 0.064 0.314 0.3   0.266 0.294 0.288 0.301
 0.3   0.301 0.32  0.045 0.009 0.264 0.039 0.008 0.324 0.052 0.01  0.307
 0.052 0.01  0.724 0.398 0.125 0.708 0.397 0.118 0.216 0.057 0.453 0.166
 1.    0.328]
VertexDiversity 0.25041009327755565
EdgeDiversity 0.760387811634349

penalty=0
[0.191 0.196 0.189 0.196 0.194 0.062 0.26  0.283 0.234 0.259 0.263 0.259
 0.261 0.25  0.354 0.044 0.01  0.282 0.042 0.008 0.35  0.059 0.011 0.345
 0.055 0.01  0.505 0.346 0.122 0.469 0.361 0.127 0.221 0.057 0.416 0.146
 1.    0.273]
VertexDiversity 0.2293155782423465
EdgeDiversity 0.760387811634349

penalty=8
[0.183 0.19  0.185 0.193 0.187 0.057 0.284 0.286 0.237 0.283 0.276 0.261
 0.272 0.276 0.315 0.039 0.008 0.232 0.034 0.007 0.307 0.048 0.009 0.271
 0.047 0.008 0.672 0.352 0.109 0.646 0.359 0.102 0.192 0.051 0.402 0.151
 1.    0.289]
VertexDiversity 0.23209823814469455
EdgeDiversity 0.760387811634349
"""
