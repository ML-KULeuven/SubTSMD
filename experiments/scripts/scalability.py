
import time
import os
import toml
import common

import pandas as pd
from sub_tsmd import load, load_test, apply_sub_tsmd


def evaluate(config, tuned_parameters, benchmark_set, ds_name, meta, methods: dict, methods_subspace: dict):
    # Load the data
    Xs, _ = load_test(f'{config["data_path"]}/{benchmark_set}/{ds_name}')

    # Execute
    results = []
    for i, X in enumerate(Xs):

        for motif_discovery_name, motif_discovery_method in methods.items():
            start = time.time()
            motif_discovery_method(X, meta, **tuned_parameters[motif_discovery_name])
            total_time = time.time() - start

            results.append(pd.Series(
                {
                    'benchmark_set': benchmark_set,
                    'ds_name': ds_name,
                    'time_series_id': i,
                    'model': motif_discovery_name,
                    'time [s]': total_time
                }
            ))

        for motif_discovery_name, motif_discovery_method in methods_subspace.items():
            hyper_parameters, subspace_hyper_parameters = common.split_hyper_parameters(tuned_parameters[motif_discovery_name])

            start = time.time()
            univariate_motif_sets = motif_discovery_method(X, meta, **hyper_parameters)
            time_univariate_motif_discovery = time.time() - start

            start = time.time()
            apply_sub_tsmd(univariate_motif_sets, **subspace_hyper_parameters)
            time_subspace_motif_discovery = time.time() - start

            results.append(pd.Series(
                {
                    'benchmark_set': benchmark_set,
                    'ds_name': ds_name,
                    'time_series_id': i,
                    'model': motif_discovery_name,
                    'time motif discovery [s]': time_univariate_motif_discovery,
                    'time motif aggregation [s]': time_subspace_motif_discovery,
                    'time [s]': time_univariate_motif_discovery + time_subspace_motif_discovery
                }
            ))

    # Save the results
    path = f'{config["results_path"]}/scalability/{benchmark_set}'
    os.makedirs(path, exist_ok=True)
    pd.concat(results, axis=1).T.to_csv(f'{path}/{ds_name}.csv', index=False)


def main(config_path: str, tuned_parameters_path: str, datasets: list, methods: dict, methods_subspace: dict):

    # Read the config file
    with open(config_path, 'r') as f:
        config = toml.load(f)

    # Read the tuned parameters
    with open(tuned_parameters_path, 'r') as f:
        tuned_parameters = toml.load(f)

    # Load the metadata
    metadata, benchmark_sets = common.load_metadata(config["data_path"], datasets)

    # Run once on demonstration data, to make sure that the all numba-related stuff is compiled
    X, y = load(f'{config["data_path"]}/synthetic/demonstration/subspace.pkl')
    meta = metadata.iloc[0]  # Values don't really matter
    for motif_discovery_name, motif_discovery_method in methods.items():
        motif_discovery_method(X, meta, **tuned_parameters['synthetic']['s1'][motif_discovery_name])
    for motif_discovery_name, motif_discovery_method in methods_subspace.items():
        hyper_parameters, subspace_hyper_parameters = common.split_hyper_parameters(tuned_parameters['synthetic']['s1'][motif_discovery_name])
        univariate_motif_sets = motif_discovery_method(X, meta, **hyper_parameters)
        apply_sub_tsmd(univariate_motif_sets, **subspace_hyper_parameters)

    # Evaluate the models
    for ds_name, meta in metadata.iterrows():
        benchmark_set = benchmark_sets[ds_name]
        evaluate(config, tuned_parameters['synthetic']['s1'], benchmark_set, ds_name, meta, methods, methods_subspace)


if __name__ == '__main__':
    main(*common.parse_args(require_tuned_parameters=True))
