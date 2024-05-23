from abc import abstractclassmethod
import numpy as np
from online_bipartite_graph import OnlineBipartiteGraph
from bipartite_graph import BipartiteGraph

"""
Hard 0.823 instance of [MGS12]
See Proposition 5.3 of https://arxiv.org/pdf/1007.1673.pdf

Require n > 1000
Y_k = a random pattern amongst the (n choose k) patterns neighboring k vertices
Define alpha = c^*_{2.5}, approx 0.81034
Define m = 1/2 * alpha * n

Offline graph:
m from Y_2
m from Y_3
n-2m from Y_n, i.e. connects to all offline nodes
"""
def generate_MGS(n):
    assert n > 1000
    alpha = 0.81034
    m = int(1/2 * alpha * n)

    adjlist = []
    # Add from Y_2
    for idx in range(m):
        nbrs = np.random.choice(np.arange(n), 2, replace=False)
        adjlist.append(nbrs)
    # Add from Y_3
    for idx in range(m):
        nbrs = np.random.choice(np.arange(n), 3, replace=False)
        adjlist.append(nbrs)
    # Add from Y_n
    for idx in range(n - 2*m):
        adjlist.append(np.arange(n))
    return BipartiteGraph(size_U = n, size_V = n, graph_rep="adj_list_u", graph_rep_data=adjlist)