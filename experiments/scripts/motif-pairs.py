
import os
import toml
import argparse
import pandas as pd

from competitors import *
import common
from sub_tsmd import load_test, apply_sub_tsmd


def evaluate(config, tuned_parameters, benchmark_set, ds_name, meta, methods: list[str]):
    # Load the data
    Xs, ys = load_test(f'{config["data_path"]}/{benchmark_set}/{ds_name}')
    n_dims = int(ds_name.split('-')[0].split('=')[1])

    results = []
    for i, (X, y) in enumerate(zip(Xs, ys)):

        for method in methods:
            if method == "EMD_star":
                motif_sets = emd_star_wrapper(
                    X,
                    l_min=int(meta['l_min']),
                    l_max=int(meta['l_max']),
                    k_max=int(meta['k_max']),
                    n_dims=n_dims,
                    elbow_deviation=tuned_parameters[method]['elbow_deviation']
                )

            elif method == "LAMA":
                motif_sets = lama_wrapper(
                    X,
                    l_min=int(meta['l_min']),
                    l_max=int(meta['l_max']),
                    k_max=int(meta['k_max']),
                    n_dims=n_dims,
                    elbow_deviation=tuned_parameters[method]['elbow_deviation']
                )

            elif method == "MMotifs":
                motif_sets = mmotifs_wrapper(
                    X,
                    window_size=common.window_size(int(meta['l_min']), l_max=int(meta['l_max'])),
                    r=meta['r'],
                    max_motifs=1,
                    k=n_dims-1
                )

            elif method == "LoCoMotif":
                univariate_motif_sets = univariate_locomotif_wrapper(
                    X,
                    l_min=int(meta['l_min']),
                    l_max=int(meta['l_max']),
                    rho=tuned_parameters[method]['rho'],
                    warping=tuned_parameters[method]['warping']
                )
                motif_sets = apply_sub_tsmd(univariate_motif_sets)

            else:
                raise ValueError(f"Invalid method name given: {method}!")

            results.append(pd.Series(
                {
                    'benchmark_set': benchmark_set,
                    'ds_name': ds_name,
                    'time_series_id': i,
                    'model': method
                }
                |
                common.compute_scores(y, motif_sets, config['evaluation'])
            ))

    # Save the results
    path = f'{config["results_path"]}/test/{benchmark_set}'
    os.makedirs(path, exist_ok=True)
    pd.concat(results, axis=1).T.to_csv(f'{path}/{ds_name}.csv', index=False)


def main(config: str, tuned_parameters: str, motif_discovery: list, subspace_motif_discovery: list):

    # Read the config file
    with open(config, 'r') as f:
        config = toml.load(f)
    with open(tuned_parameters, 'r') as f:
        tuned_parameters = toml.load(f)

    # Load the metadata
    metadata, benchmark_sets = common.load_metadata(config["data_path"], ['motif-pairs/*'])

    # Format the methods
    def _methods(given: list[str], all_methods: list[str]) -> list[str]:
        if len(given) == 1 and given[0] == 'none':
            return []
        elif len(given) == 1 and given[0] == 'all':
            return all_methods
        else:
            return given
    methods = _methods(motif_discovery, ['EMD_star', 'LAMA', 'MMotifs']) + _methods(subspace_motif_discovery, ['LoCoMotif'])

    # Run the experiments
    for ds_name, meta in metadata.iterrows():
        print(f'{"-"*50}\n{ds_name}\n{"-"*50}\n')
        benchmark_set = benchmark_sets[ds_name]
        evaluate(config, tuned_parameters['synthetic']['s4'], benchmark_set, ds_name, meta, methods)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to run models on datasets with a configuration file.")
    parser.add_argument('--config', type=str, required=True, help=f'Path to the configuration file')
    parser.add_argument('--tuned_parameters', type=str, required=True, help=f'Path to the configuration file with tuned hyperparameters')
    parser.add_argument('--motif_discovery', type=str, nargs='+', required=True, help="List of models to use.\nValid options: {'EMD_star', 'LAMA', 'MMotifs'}")
    parser.add_argument('--subspace_motif_discovery', type=str, nargs='+', required=True, help="List of models to use.\nValid options: {'LoCoMotif'}")
    main(**vars(parser.parse_args()))
