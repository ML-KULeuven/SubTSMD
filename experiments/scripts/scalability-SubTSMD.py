import os
import random
import argparse
import toml
import time
import tqdm

import pandas as pd
import numpy as np
from sub_tsmd import apply_sub_tsmd


########################################################################
# GENERATING MOTIF SETS
########################################################################

def generate_k_integers_with_sum(K, N, rng):
    # Step 1: Generate K random values
    random_values = rng.uniform(size=K)

    # Step 2: Normalize and scale to sum up to N
    scaled_values = random_values / random_values.sum() * N

    # Step 3: Round to nearest integers
    integer_values = np.round(scaled_values).astype(int)

    # Adjust the sum to be exactly N
    current_sum = integer_values.sum()
    difference = N - current_sum

    # Distribute the difference
    for i in range(abs(difference)):
        if difference > 0:
            integer_values[i % K] += 1
        elif difference < 0:
            integer_values[i % K] -= 1

    return integer_values


def generate_motifs(coverage, k, l_min, l_max, seed):
    rng = np.random.default_rng(seed)

    motif_lengths = [rng.integers(l_min, l_max) for _ in range(k)]
    total_motif_length = sum(motif_lengths)
    total_length = int(total_motif_length / coverage)
    uncovered = total_length - total_motif_length

    distances_between_motifs = generate_k_integers_with_sum(k, uncovered, rng)
    start_positions = np.cumsum(distances_between_motifs) + np.cumsum(motif_lengths) - motif_lengths[0]

    motifs = [(start_position, start_position + motif_length) for motif_length, start_position in zip(motif_lengths, start_positions)]

    return motifs


def partition_motifs(motifs, nb_motif_sets):
    # Ensure there are enough elements to form k sets with at least 2 elements each
    if len(motifs) < 2 * nb_motif_sets:
        raise ValueError("Not enough elements to form k sets with at least 2 elements each.")

    # Shuffle the elements randomly
    random.shuffle(motifs)

    # Initialize the sets
    sets = [[] for _ in range(nb_motif_sets)]

    # Distribute the elements into the sets
    for i in range(len(motifs)):
        sets[i % nb_motif_sets].append(motifs[i])

    return sets


def format_motif_set(motif_set, attribute, nb_attributes):
    return np.array([i == attribute for i in range(nb_attributes)]), np.expand_dims(np.array(motif_set), axis=-1)


def generate(coverage, D, nb_motifs_per_attribute, nb_motif_sets_per_attribute, l_min, l_max, seed):
    univariate_motifs = []
    for d in range(D):
        motifs = generate_motifs(coverage, nb_motifs_per_attribute, l_min, l_max, seed + d)
        motif_sets = partition_motifs(motifs, nb_motif_sets_per_attribute)
        univariate_motifs.append([format_motif_set(motif_set, d, D) for motif_set in motif_sets])
    return univariate_motifs


########################################################################
# MAIN SCRIPT
########################################################################


def apply_job(seed, repeat, coverage, threshold, linkage, dimension, nb_motifs, nb_motif_sets, full_numba, l_min, l_max):
    base = {
        'coverage': coverage,
        'dimension': dimension,
        'threshold': threshold,
        'linkage': linkage,
        'nb_motifs': nb_motifs,
        'nb_motif_sets': nb_motif_sets,
        'full_numba': full_numba
    }
    tqdm.tqdm.write(str(base))

    results = []
    for i in range(repeat):
        motif_sets = generate(
            coverage=coverage,
            D=dimension,
            nb_motifs_per_attribute=nb_motifs,
            nb_motif_sets_per_attribute=nb_motif_sets,
            l_min=l_min,
            l_max=l_max,
            seed=seed+i
        )

        start = time.perf_counter_ns()
        apply_sub_tsmd(motif_sets, threshold, linkage)
        total_time_ns = time.perf_counter_ns() - start

        results.append(pd.Series(base | {'repeat': i, 'Time [ms]': total_time_ns / 1e6}))

    return pd.concat(results, axis=1).T


def main():
    parser = argparse.ArgumentParser(description="Script to run models on datasets with a configuration file.")
    parser.add_argument('--config', type=str, required=True, help=f'Path to the configuration file')
    args = parser.parse_args()

    # Read the important part for this experiment from the config file
    with open(args.config, 'r') as f:
        full_config = toml.load(f)

    # Extract relevant information from the config
    results_path = full_config['results_path']
    config = full_config['scalability-sub-tsmd']

    # To compile the numba code
    print('Prerun...')
    motif_sets = generate(1000, 5, 20, 4, 25, 35, 0)
    apply_sub_tsmd(motif_sets)

    print('Initializing the jobs...')
    jobs = []
    for coverage in config['coverage']:
        for threshold in config['delta']:
            for linkage in config['linkage']:

                for dimension in config['dimension']:
                    jobs.append((coverage, threshold, linkage, dimension, config['default_nb_motifs'], config['default_nb_motif_sets'], True))

                for nb_motifs in config['nb_motifs']:
                    # Only use one motif set per attribute, because otherwise maybe the motifs can not be divided over the motif sets
                    jobs.append((coverage, threshold, linkage, config['default_dimension'], nb_motifs, 1, True))

                for nb_motif_sets in config['nb_motif_sets']:
                    jobs.append((coverage, threshold, linkage, config['default_dimension'], config['default_nb_motifs'], nb_motif_sets, True))

    # jobs = []  # To purely evaluate delta
    # for coverage in config['coverage']:
    #     for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #         jobs.append((coverage, threshold, "average", config['default_dimension'], config['default_nb_motifs'], config['default_nb_motif_sets'], True))

    print('Running jobs...')
    results = pd.concat([apply_job(config['seed'], config['repeat'], *job, config['l_min'], config['l_max']) for job in tqdm.tqdm(jobs)])

    print('Saving the results...')
    if not os.path.exists(f'{results_path}/scalability'):
        os.makedirs(f'{results_path}/scalability')
    results.to_csv(f'{results_path}/scalability/SubTSMD.csv', index=False)


if __name__ == '__main__':
    main()
