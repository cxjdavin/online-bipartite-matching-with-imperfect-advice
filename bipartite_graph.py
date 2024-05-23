import igraph as ig
import numpy as np

from typing import Dict, Tuple, Set, Union, Callable, List

class BipartiteGraph(object):
    '''
    graph_rep, graph_rep_data can be any of the following,
        1) adj_list_u : List[List[int]] adjacency list of indices in V
        2) adj_list_v : List[List[int]] adjacency list of indices in U
        3) edge_list: List[Tuple(int, int)]
    '''
    def __init__(self, size_U: int, size_V: int, 
                 graph_rep: str, graph_rep_data,
                 augment:bool = True):
        self.size_U, self.size_V = size_U, size_V
        
        # Maintain 3 representations for graph internally for convenience.
        # Adjacency list
        self.neighbors_U = [[] for i in range(size_U)]
        self.neighbors_V = [[] for i in range(size_V)]
        # Edge list list of the form (u_idx, v_idx), where u->v
        self.edge_list = []
        
        # Generate graph based on whichever graph representation was given to us.
        if graph_rep == 'adj_list_u' or graph_rep == 'adj_list_U':
            adj_list_U = graph_rep_data
            assert len(adj_list_U) == size_U
            for idx_u, adj_to_u in enumerate(adj_list_U):
                for idx_v in adj_to_u:
                    self.neighbors_U[idx_u].append(idx_v)
                    self.neighbors_V[idx_v].append(idx_u)
                    self.edge_list.append((idx_u, idx_v))
        elif graph_rep == 'adj_list_v' or graph_rep == 'adj_list_V':
            adj_list_V = graph_rep_data
            assert len(adj_list_V) == size_V
            for idx_v, adj_to_v in enumerate(adj_list_V):
                for idx_u in adj_to_v:
                    self.neighbors_U[idx_u].append(idx_v)
                    self.neighbors_V[idx_v].append(idx_u)
                    self.edge_list.append((idx_u, idx_v))
        elif graph_rep == 'edge_list':
            assert False, 'Not implemented'

        # Sort adjacency lists such that they are ordered incrementally
        for adj_to_v in self.neighbors_V:
            adj_to_v.sort()
        for adj_to_u in self.neighbors_U:
            adj_to_u.sort()

        # Enforce there are no parallel edges
        self.check_parallel_edges()

        # Augment with dummy singletons such that there are equally many u and v vertices.
        if augment:
            self.make_uv_equal_size()

    def get_size_u(self):
        return self.size_U
    
    def get_size_v(self):
        return self.size_V

    def get_neighbors_v(self, idx_v):
        return self.neighbors_V[idx_v]

    def get_neighbors_u(self, idx_u):
        return self.neighbors_U[idx_u]

    def get_num_edges(self):
        return len(self.edge_list)

    def make_uv_equal_size(self):
        while self.size_U < self.size_V:
            self.neighbors_U.append([])
            self.size_U += 1
        while self.size_V < self.size_U:
            self.neighbors_V.append([])
            self.size_V += 1

    def is_uv_equal_size(self):
        return self.size_U == self.size_V 

    def check_parallel_edges(self):
        for adj_to_u in self.neighbors_U:
            assert len(set(adj_to_u)) == len(adj_to_u)

    # https://python.igraph.org/en/0.11.3/tutorials/bipartite_matching.html
    def maximum_matching(self):
        edges = []
        for u,v in self.edge_list:
            edges.append((u, self.get_size_u() + v))
        g = ig.Graph.Bipartite([0] * self.get_size_u() + [1] * self.get_size_v(), edges)
        assert g.is_bipartite()
        matching = g.maximum_bipartite_matching()
        u_matched = [None] * self.get_size_u()
        v_matched = [None] * self.get_size_v()
        edges_matched = []
        for u_idx in range(self.get_size_u()):
            if matching.is_matched(u_idx):
                v_idx = matching.match_of(u_idx) - self.get_size_u()
                u_matched[u_idx] = v_idx
                v_matched[v_idx] = u_idx
                edges_matched.append((u_idx, v_idx))
        return u_matched, v_matched, edges_matched
