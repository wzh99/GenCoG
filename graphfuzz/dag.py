import numpy as np
from numpy.random import Generator

from muffin.dag import Node


def rn_model(num_nodes: int, p: float, rng: Generator):
    # Initialize nodes
    nodes = [Node(0, is_input=True)]
    for i in range(num_nodes):
        node = Node(i + 1)
        nodes.append(node)
    nodes[-1].is_output = True

    # Randomly connect to nodes after each node
    for i, node in enumerate(nodes[:-1]):
        node.connect_to([nodes[i + 1]])
        if rng.random() < p:
            node.connect_to([nodes[rng.integers(i + 1, num_nodes + 1)]])

    return nodes


def ws_model(n: int, k: int, p: float, rng: Generator):
    """
    Adapted from https://github.com/seungwonpark/RandWireNN/blob/master/model/graphs/ws.py
    """
    adj = np.eye(n, dtype=bool)

    # Initial connection
    for i in range(n):
        for j in range(i - k // 2, i + k // 2 + 1):
            real_j = j % n
            if real_j == i:
                continue
            adj[real_j, i] = adj[i, real_j] = True

    # Rewire
    for i in range(n):
        for j in range(k // 2):
            current = (i + j + 1) % n
            if rng.random() < p:  # rewire
                unoccupied = [x for x in range(n) if not adj[i, x]]
                rewired = np.random.choice(unoccupied)
                adj[i, current] = adj[current, i] = False
                adj[i, rewired] = adj[rewired, i] = True

    # Find all edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append((i, j))
    edges.sort()

    # Create and connect nodes
    nodes = [Node(i) for i in range(n + 2)]
    nodes[0].is_input = True
    nodes[-1].is_output = True
    for u, v in edges:
        nodes[u + 1].connect_to([nodes[v + 1]])
    for node in nodes:
        if node.id == 0 or node.id == n + 1:
            continue
        if len(node.inbound_nodes) == 0:
            nodes[0].connect_to([node])
        if len(node.outbound_nodes) == 0:
            node.connect_to([nodes[-1]])

    return nodes
