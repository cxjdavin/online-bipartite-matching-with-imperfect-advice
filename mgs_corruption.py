import copy
import matplotlib.pyplot as plt
import numpy as np
import sys

from collections import defaultdict
from online_bipartite_graph import OnlineBipartiteGraph
from mgs12 import generate_MGS
from online_algs import Ranking, TestAndMatch
from tqdm import tqdm

n = int(sys.argv[1])
num_times = int(sys.argv[2])
bucket_threshold = int(sys.argv[3])
assert n > 1000
corruption_type = int(sys.argv[4])
print("n: {0}, num_times = {1}, bucket_threshold = {2}, corruption_type = {3}".format(n, num_times, bucket_threshold, corruption_type))

# Collect statistics for plotting
all_true_L1 = defaultdict(list)
all_ranking_cr = defaultdict(list)
all_tam_cr = defaultdict(list)
all_tam_no_patch_cr = defaultdict(list)
all_tam_no_sigma_cr = defaultdict(list)
all_tam_no_bucket_cr = defaultdict(list)

rng_seed = 314159
instance_rng = np.random.RandomState(rng_seed)
for _ in tqdm(range(num_times), desc="Num times", leave=False):
    input_graph = generate_MGS(n)
    for alpha_idx in tqdm(range(11), desc="alpha"):
        alpha = alpha_idx / 10
        instance_seed = instance_rng.randint(1e6)
        instance = OnlineBipartiteGraph(input_graph, random_arrival_seed=instance_seed)

        #
        # Run Ranking
        #
        # print("alpha_idx = {0} | Running Ranking".format(alpha_idx))
        instance = OnlineBipartiteGraph(input_graph, random_arrival_seed=instance_seed)
        ranking = Ranking(instance, rng_seed)
        ranking.solve()
        ranking_cr = ranking.report_for_plot()

        #
        # Setup c_hat and corrupt
        #
        instance = OnlineBipartiteGraph(input_graph, random_arrival_seed=instance_seed)
        beta = 0.696
        c_hat = instance.get_c_star()
        
        # Map c_hat patterns to RANDOM indices. Random is crucial because we are in random arrival model
        shuffled_indices = np.arange(n)
        np.random.shuffle(shuffled_indices)
        idx_to_pattern = dict()
        idx = 0
        for pattern, count in c_hat.items():
            for _ in range(count):
                idx_to_pattern[shuffled_indices[idx]] = pattern
                idx += 1

        # Perform corruption on alpha fraction of online vertices
        num_corrupted = int(alpha * n)
        corrupted = np.random.choice(n, num_corrupted, replace = False)
        corrupted_indices = shuffled_indices[corrupted]
        assert len(set(corrupted_indices)) == num_corrupted
        for idx in corrupted_indices:
            original_pattern = idx_to_pattern[idx]
            c_hat[original_pattern] -= 1
            if c_hat[original_pattern] == 0:
                del c_hat[original_pattern]
            edge_prob = 0.1 * np.log(n)/n
            new_subset = np.arange(n)[np.random.choice([True, False], n, p=[edge_prob, 1 - edge_prob])]
            if corruption_type == 1: # Add random edges
                new_pattern = original_pattern.union(new_subset)
            elif corruption_type == 2: # Set pattern to new random pattern
                new_pattern = new_subset
            else:
                assert False
            c_hat[frozenset(new_pattern)] += 1

        # Compute L1(p^*, q)
        true_L1 = instance._get_true_L1(c_hat)
        assert 0 <= true_L1 and true_L1 <= 2

        #
        # Run TestAndMatch
        #
        # print("alpha_idx = {0} | Running TaM".format(alpha_idx))
        params = {
             "bucket_threshold": bucket_threshold,
             "sigma_mapping": 1,
             "patch": 1
        }
        instance = OnlineBipartiteGraph(input_graph, random_arrival_seed=instance_seed)
        tam = TestAndMatch(instance, Ranking, beta, copy.deepcopy(c_hat), params, rng_seed=rng_seed, taken=[])
        mimic_all_the_way = tam.solve()
        tam_cr = tam.report_for_plot()
        # print(mimic_all_the_way)

        #
        # Run TestAndMatch without patch
        #
        # print("alpha_idx = {0} | Running TaM without patch".format(alpha_idx))
        params = {
             "bucket_threshold": bucket_threshold,
             "sigma_mapping": 1,
             "patch": 0
        }
        instance = OnlineBipartiteGraph(input_graph, random_arrival_seed=instance_seed)
        tam_no_patch = TestAndMatch(instance, Ranking, beta, copy.deepcopy(c_hat), params, rng_seed=rng_seed, taken=[])
        mimic_all_the_way_no_patch = tam_no_patch.solve()
        tam_no_patch_cr = tam_no_patch.report_for_plot()
        # print(mimic_all_the_way_no_patch)

        #
        # Run TestAndMatch without sigma mapping
        #
        # print("alpha_idx = {0} | Running TaM without sigma mapping".format(alpha_idx))
        params = {
             "bucket_threshold": bucket_threshold,
             "sigma_mapping": 0,
             "patch": 1
        }
        instance = OnlineBipartiteGraph(input_graph, random_arrival_seed=instance_seed)
        tam_no_sigma = TestAndMatch(instance, Ranking, beta, copy.deepcopy(c_hat), params, rng_seed=rng_seed, taken=[])
        mimic_all_the_way_no_sigma = tam_no_sigma.solve()
        tam_no_sigma_cr = tam_no_sigma.report_for_plot()
        # print(mimic_all_the_way_no_sigma)

        #
        # Run TestAndMatch without bucketing
        #
        # print("alpha_idx = {0} | Running TaM without bucketing".format(alpha_idx))
        params = {
             "bucket_threshold": 0,
             "sigma_mapping": 1,
             "patch": 1
        }
        instance = OnlineBipartiteGraph(input_graph, random_arrival_seed=instance_seed)
        tam_no_bucket = TestAndMatch(instance, Ranking, beta, copy.deepcopy(c_hat), params, rng_seed=rng_seed, taken=[])
        mimic_all_the_way_no_bucket = tam_no_bucket.solve()
        tam_no_bucket_cr = tam_no_bucket.report_for_plot()
        # print(mimic_all_the_way_no_bucket)

        # Collect results
        all_true_L1[alpha_idx].append(true_L1)
        all_ranking_cr[alpha_idx].append(ranking_cr)
        all_tam_cr[alpha_idx].append(tam_cr)
        all_tam_no_patch_cr[alpha_idx].append(tam_no_patch_cr)
        all_tam_no_sigma_cr[alpha_idx].append(tam_no_sigma_cr)
        all_tam_no_bucket_cr[alpha_idx].append(tam_no_bucket_cr)

        # print(true_L1, ranking_cr, tam_cr, tam_no_patch_cr, tam_no_sigma_cr, tam_no_bucket_cr)

# print("Plotting graph...")

# Compute error bars
mean_true_L1 = [np.mean(all_true_L1[alpha_idx]) for alpha_idx in range(11)]
mean_ranking_cr = [np.mean(all_ranking_cr[alpha_idx]) for alpha_idx in range(11)]
mean_tam_cr = [np.mean(all_tam_cr[alpha_idx]) for alpha_idx in range(11)]
mean_tam_no_patch_cr = [np.mean(all_tam_no_patch_cr[alpha_idx]) for alpha_idx in range(11)]
mean_tam_no_sigma_cr = [np.mean(all_tam_no_sigma_cr[alpha_idx]) for alpha_idx in range(11)]
mean_tam_no_bucket_cr = [np.mean(all_tam_no_bucket_cr[alpha_idx]) for alpha_idx in range(11)]

std_true_L1 = [np.std(all_true_L1[alpha_idx]) for alpha_idx in range(11)]
std_ranking_cr = [np.std(all_ranking_cr[alpha_idx]) for alpha_idx in range(11)]
std_tam_cr = [np.std(all_tam_cr[alpha_idx]) for alpha_idx in range(11)]
std_tam_no_patch_cr = [np.std(all_tam_no_patch_cr[alpha_idx]) for alpha_idx in range(11)]
std_tam_no_sigma_cr = [np.std(all_tam_no_sigma_cr[alpha_idx]) for alpha_idx in range(11)]
std_tam_no_bucket_cr = [np.std(all_tam_no_bucket_cr[alpha_idx]) for alpha_idx in range(11)]

fig = plt.figure()
plt.errorbar(np.arange(11)/10, mean_ranking_cr, yerr=std_ranking_cr, label="Ranking", marker='x')
plt.errorbar(np.arange(11)/10, mean_tam_cr, yerr=std_tam_cr, label="TaM", marker='+')
plt.errorbar(np.arange(11)/10, mean_tam_no_patch_cr, yerr=std_tam_no_patch_cr, label="TaM w/o patch", marker='s', markerfacecolor='none')
plt.errorbar(np.arange(11)/10, mean_tam_no_sigma_cr, yerr=std_tam_no_sigma_cr, label="TaM w/o sigma", marker='*')
plt.errorbar(np.arange(11)/10, mean_tam_no_bucket_cr, yerr=std_tam_no_bucket_cr, label="TaM w/o bucket", marker='o', markerfacecolor='none')
plt.legend()
plt.ylabel("Competitive ratios")
plt.xlabel("Fraction of corrupted vertices")
plt.title("n = {0}, {1} runs, corruption type: {2}".format(n, num_times, corruption_type))
plt.savefig("n={0},run={1},type={2}".format(n, num_times, corruption_type))
