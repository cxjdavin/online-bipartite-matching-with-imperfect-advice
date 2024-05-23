import copy
import numpy as np

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Tuple, Set, List

from online_bipartite_graph import OnlineBipartiteGraph
from bipartite_graph import BipartiteGraph

"""
Our OnlineAlgorithm class
"""
class OnlineAlgorithm(ABC):
    def __init__(self,
            instance: OnlineBipartiteGraph,
            rng_seed: int = 314159,
            taken: List[int] = []):
        self.instance = instance
        self.n = instance.n
        self.done = False
        self.observed = []
        self.rng = np.random.RandomState(rng_seed)

        # Extra information to allow running on half-run instances
        self.taken = taken

    def report_for_plot(self):
        matches = self.instance.get_matching_so_far()
        num_matches = len([x for x in matches if x is not None])
        competitive_ratio = num_matches / self.instance._n_star
        return competitive_ratio

    @abstractmethod
    def solve(self) -> None:
        pass

"""
KVV's Ranking algorithm

When vertex is taken, set rank to n+1 so it will never be argmin
"""
class Ranking(OnlineAlgorithm):
    def solve(self) -> None:
        assert self.done is False

        # Setup offline vertex rankings
        offline_ranking = np.arange(self.n)
        self.rng.shuffle(offline_ranking)
        for u in self.taken:
            offline_ranking[u] = self.n + 1

        # Run Ranking
        while not self.instance.is_done():
            v, arrival = self.instance.get_next_arrival()
            nbrs = list(arrival)
            ranks = [offline_ranking[u] for u in nbrs]
            if len(ranks) == 0 or min(ranks) > self.n:
                assert self.instance.attempt_matching(v, None)
            else:
                chosen_u = nbrs[np.argmin(ranks)]
                assert offline_ranking[chosen_u] <= self.n
                offline_ranking[chosen_u] = self.n + 1
                assert self.instance.attempt_matching(v, chosen_u)
            self.observed.append((v, arrival))

        assert self.instance.get_next_arrival() is None
        self.done = True

class TestAndMatch(OnlineAlgorithm):
    def __init__(self,
            instance: OnlineBipartiteGraph,
            baselineObj: OnlineAlgorithm,
            beta: float,
            original_c_hat: Dict[frozenset[int], int],
            params: Dict[str, int] = None,
            rng_seed: int = 314159,
            taken: List[int] = []) -> None:
        super().__init__(instance, rng_seed, taken)
        self.baselineObj = baselineObj
        self.beta = beta

        # Setup algo mode
        # print(params)
        if "sigma_mapping" in params.keys():
            self.sigma_mapping = params["sigma_mapping"]
        else:
            self.sigma_mapping = 0 # No sigma mapping

        if "bucket_threshold" in params.keys():
            self.bucket_threshold = params["bucket_threshold"]
        else:
            self.bucket_threshold = 0 # No bucketing

        self.original_c_hat = original_c_hat
        self.patched_c_hat = copy.deepcopy(original_c_hat)
        self.compute_mimic_matching()
        if "patch" in params.keys() and params["patch"] == 1 and self.n_hat < self.n:
            self.patch()
            self.compute_mimic_matching()
            assert self.n_hat == self.n
        
        # Setup c_hat (accounting for possible patching and bucketing)
        self.c_hat = defaultdict(int)
        self.pattern_to_c_hat_idx = defaultdict(int)
        self.c_hat_idx_to_pattern = defaultdict(list)
        self.small_patterns = []
        self.small_counts = 0
        for pattern, count in self.patched_c_hat.items():
            assert count > 0
            if count <= self.bucket_threshold:
                self.small_patterns.append(pattern)
                self.small_counts += count
            else:
                pattern_idx = len(self.pattern_to_c_hat_idx)
                self.pattern_to_c_hat_idx[pattern] = pattern_idx
                self.c_hat_idx_to_pattern[pattern_idx] = [pattern]
                self.c_hat[self.pattern_to_c_hat_idx[pattern]] = count
        if self.small_counts > 0:
            bucket_idx = len(self.pattern_to_c_hat_idx)
            self.c_hat_idx_to_pattern[bucket_idx] = []
            for pattern in self.small_patterns:
                self.pattern_to_c_hat_idx[pattern] = bucket_idx
                self.c_hat_idx_to_pattern[bucket_idx].append(pattern)
            self.c_hat[bucket_idx] = self.small_counts
        assert sum(self.c_hat.values()) == self.n

        # Setup sample size and testing threshold
        self.eps = self.n_hat/self.n - self.beta
        self.r = len(self.c_hat.keys())
        self.num_samples = int(self.r / pow(self.eps, 2))
        self.testing_threshold = 2 * (self.n_hat / self.n - self.beta) - self.eps
        # print('r:', self.r, ', bucket:', self.bucket_threshold, ', threshold:', self.testing_threshold)

    def patch(self):
        # Identify unmatched vertices
        unmatched_offline = []
        unmatched_online = []
        for u in range(self.n):
            if self.u_matched[u] is None:
                unmatched_offline.append(u)
        for v in range(self.n):
            if self.v_matched[v] is None:
                unmatched_online.append(v)
        assert len(unmatched_offline) == len(unmatched_online)
        assert self.n_hat == self.n - len(unmatched_online)

        # Create complete bipartite between these vertices
        v_idx = 0
        for pattern, count in self.original_c_hat.items():
            for _ in range(count):
                if v_idx in unmatched_online:
                    self.patched_c_hat[pattern] -= 1
                    if self.patched_c_hat[pattern] == 0:
                        del self.patched_c_hat[pattern]
                    new_pattern = frozenset(pattern.union(unmatched_offline))
                    self.patched_c_hat[new_pattern] += 1
                v_idx += 1

    def mimic(self) -> None:
        new_arrival = self.instance.get_next_arrival()
        assert new_arrival is not None
        v, pattern = new_arrival

        if self.sigma_mapping:
            # Map online arrival pattern to some advice pattern if possible
            possible_replacement_patterns = []
            if pattern in self.pattern_to_c_hat_idx.keys() and len(self.M_hat_matches[pattern]) > 0:
                possible_replacement_patterns = [(len(pattern), len(self.M_hat_matches[pattern]), pattern)]
            else:
                for idx, _ in self.c_hat.items():
                    for c_hat_pattern in self.c_hat_idx_to_pattern[idx]:
                        c_hat_matches = self.M_hat_matches[c_hat_pattern]
                        remaining_match_counts = len(c_hat_matches)
                        if len(c_hat_pattern.difference(pattern)) == 0 and remaining_match_counts > 0:
                            # c_hat_pattern is subset of pattern and there is remaining match count
                            pattern_size = len(c_hat_pattern)
                            possible_replacement_patterns.append((pattern_size, remaining_match_counts, c_hat_pattern))

            if len(possible_replacement_patterns) > 0:
                # Pick the largest pattern followed by highest remaining count
                _, _, c_hat_pattern = sorted(possible_replacement_patterns, reverse=True)[0]
                c_hat_matches = self.M_hat_matches[c_hat_pattern]
                self.observed.append((v, c_hat_pattern))

                # Pick any and remove from M_hat_matches
                chosen_u = list(pattern.intersection(c_hat_matches))[0]
                self.M_hat_matches[c_hat_pattern].remove(chosen_u)
                assert chosen_u not in self.taken
                self.taken.append(chosen_u)
                assert self.instance.attempt_matching(v, chosen_u)
            else:
                self.observed.append((v, pattern))
                assert self.instance.attempt_matching(v, None)
        else:
            self.observed.append((v, pattern))
            if pattern in self.pattern_to_c_hat_idx.keys() and len(self.M_hat_matches[pattern]) > 0:
                # Pick any and remove from M_hat_matches
                chosen_u = self.M_hat_matches[pattern].pop()
                assert chosen_u not in self.taken
                self.taken.append(chosen_u)
                assert self.instance.attempt_matching(v, chosen_u)
            else:
                assert self.instance.attempt_matching(v, None)

    def is_valid_probability_vector(self, p, d):
        return len(p) == d and np.isclose(sum(p), 1) and np.min(p) >= 0 and np.max(p) <= 1

    # Modifies in-place: self.u_match, self.v_matched, self.M_hat_matches, self.n_hat
    def compute_mimic_matching(self):
        adj_list_v = []
        for pattern, count in self.patched_c_hat.items():
            list_edges = [x for x in pattern]
            for _ in range(count):
                adj_list_v.append(list_edges) # We can use a shallow copy here.
        G = BipartiteGraph(self.n, self.n, 'adj_list_v', adj_list_v)
        G.check_parallel_edges()
        assert G.is_uv_equal_size()

        u_matched, v_matched, edges_matched = G.maximum_matching()
        self.u_matched = u_matched
        self.v_matched = v_matched

        self.M_hat_matches = defaultdict(set)
        u_taken = set()
        v_idx = 0
        for pattern, count in self.patched_c_hat.items():
            for _ in range(count):
                if v_matched[v_idx] is not None:
                    assert v_matched[v_idx] not in u_taken
                    self.M_hat_matches[pattern].add(v_matched[v_idx])
                    u_taken.add(v_matched[v_idx])
                v_idx += 1
        self.n_hat = len(edges_matched)

    def simulateP(self, num_samples: int) -> Tuple[frozenset[int]]:
        assert num_samples <= len(self.observed)
        samples = []
        counter = 0
        while len(samples) < num_samples:
            if np.random.random() < counter / self.n:
                _, arrival = self.observed[np.random.randint(counter)]
            else:
                _, arrival = self.observed[counter]
                counter += 1
            samples.append(arrival)
        return tuple(samples)

    def L1_tester(self) -> bool:
        d = len(self.c_hat.keys())+1
        assert d == self.r + 1

        # Form vector q
        q = [0] * d
        for idx, counts in self.c_hat.items():
            q[idx] = counts / self.n
        assert q[d-1] == 0 # 0 in "rest"
        assert self.is_valid_probability_vector(q, d)

        samples = self.simulateP(self.num_samples)
        p_hat = []
        for pattern in samples:
            sampled_vector = [0] * d
            if pattern in self.pattern_to_c_hat_idx.keys():
                sampled_vector[self.pattern_to_c_hat_idx[pattern]] = 1
            else:
                sampled_vector[d-1] = 1 # "rest"
            p_hat.append(sampled_vector)
        p_hat = np.sum(np.array(p_hat), axis=0) / len(samples)
        assert self.is_valid_probability_vector(p_hat, d)

        self.L1_hat = sum([abs(p_hat[i] - q[i]) for i in range(d)])
        return self.L1_hat < self.testing_threshold

    def solve(self) -> bool:
        assert self.done is False
        mimic_all_the_way = False
        if self.n_hat <= self.n * self.beta:
            # print("Calling baseline because n-hat/n <= beta, n-hat={0}, n={1}, n-hat/n={2}".format(self.n_hat, self.n, self.n_hat/self.n))
            baseline = self.baselineObj(self.instance, taken=[])
            baseline.solve()
            self.observed += baseline.observed
        elif self.num_samples > self.n:
            # print("Calling baseline because {0} > {1}".format(self.num_samples, self.n))
            baseline = self.baselineObj(self.instance, taken=[])
            baseline.solve()
            self.observed += baseline.observed
        else:
            # Run Mimic for num_samples steps to gather observations
            for _ in range(self.num_samples):
                self.mimic()
            assert len(self.observed) == self.num_samples

            # Test and decide whether to Mimic or Baseline
            if self.L1_tester():
                # Run Mimic for the remaining arrivals
                mimic_all_the_way = True
                while not self.instance.is_done():
                    self.mimic()
            else:
                # Run Baseline for the remaining arrivals
                # print("Calling baseline because failed L1 test")
                baseline = self.baselineObj(self.instance, taken=self.taken)
                baseline.solve()
                self.observed += baseline.observed

        assert self.instance.get_next_arrival() is None
        self.done = True
        return mimic_all_the_way
