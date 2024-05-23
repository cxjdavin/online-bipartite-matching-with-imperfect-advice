import copy
import numpy as np

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Tuple, Set, List
import time
from bipartite_graph import BipartiteGraph

"""
Our OnlineBipartiteGraph class

U_i and bipartite == 0 are offline nodes
V_j and bipartite == 1 are online nodes
"""
class OnlineBipartiteGraph:
    def __init__(self, input_graph: BipartiteGraph, 
                arrival_order: List[int] = None,
                random_arrival_seed: int = None
                ) -> None:
        assert input_graph.is_uv_equal_size()
        n = input_graph.get_size_u()
        self.n = n
        self._G = input_graph

        _, _, edges_matched = input_graph.maximum_matching()
        self._n_star = len(edges_matched)
        
        self._extract_patterns()
        self._mate = np.full(self.n, None)

        if arrival_order is None:
            arrival_order = np.arange(input_graph.get_size_v())
        if random_arrival_seed is None:
            np.random.shuffle(arrival_order)
        else:
            rng = np.random.RandomState()
            rng.seed(random_arrival_seed)
            rng.shuffle(arrival_order)
        self._arrival_order = arrival_order
        self._arrival_idx = 0

    """
    Extract and store individual patterns and compute c_star
    """
    def _extract_patterns(self) -> None:
        patterns = defaultdict(frozenset)
        c_star = defaultdict(int)

        for idx_v in range(self._G.get_size_v()):
            adj_to_v = self._G.get_neighbors_v(idx_v)
            v_pattern = frozenset(adj_to_v)
            patterns[idx_v] = v_pattern
            c_star[v_pattern] += 1
        self._patterns = patterns
        self._c_star = c_star

    def get_c_star(self):
        return copy.deepcopy(self._c_star)

    """
    Returns _G

    Warning: Only for OPT and/or evaluation purposes
    """
    def _get_true_graph(self) -> BipartiteGraph:
        return self._G

    """
    Returns 0 <= L1(c_star, c_hat) <= n

    Warning: Only for OPT and/or evaluation purposes
    """
    def _get_true_L1(self, c_hat: Dict[frozenset[int], int]) -> int:
        assert sum(c_hat.values()) == self.n
        L1 = 2 * self.n
        common_patterns = set(self._c_star.keys()).intersection(c_hat.keys())
        for label in common_patterns:
            L1 -= 2 * min(self._c_star[label], c_hat[label])
        return L1 / self.n

    """
    Returns whether there are still incoming vertices
    """
    def is_done(self) -> bool:
        return self._arrival_idx == self.n

    """
    Simulate an online arrival
    """
    def get_next_arrival(self) -> Tuple[int, frozenset[int]]:
        if self._arrival_idx == self.n:
            return None
        else:
            v = self._arrival_order[self._arrival_idx]
            self._arrival_idx += 1
            return (v, self._patterns[v])

    """
    Attempts to match online vertex v with offline vertex u
    """
    def attempt_matching(self, v: int, u: int) -> bool:
        assert v is not None
        assert 0 <= v and v < self.n
        if u is not None:
            assert 0 <= u and u < self.n
            if self._mate[u] is not None:
                return False
            self._mate[u] = v
        return True

    """
    Returns a list of mates for offline vertices made so far
    _mate[u] = None means u is unmatched
    """
    def get_matching_so_far(self) -> np.ndarray:
        return self._mate
