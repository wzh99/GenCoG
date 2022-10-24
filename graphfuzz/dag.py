from numpy.random import Generator

from muffin.dag import Node

RN_PROB = 0.5


def rn_model(num_nodes: int, rng: Generator):
    # Initialize nodes
    nodes = [Node(0, is_input=True)]
    for i in range(num_nodes):
        node = Node(i + 1)
        nodes.append(node)
    nodes[-1].is_output = True

    # Randomly connect to nodes after each node
    for i, node in enumerate(nodes[:-1]):
        node.connect_to([nodes[i + 1]])
        if rng.random() < RN_PROB:
            node.connect_to([nodes[rng.integers(i + 1, num_nodes + 1)]])

    return nodes
