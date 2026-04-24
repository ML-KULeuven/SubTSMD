
import os
import toml
import pandas as pd

from sub_tsmd import load_test, apply_sub_tsmd
import common


def evaluate(config, tuned_parameters, benchmark_set, ds_name, meta, methods: dict, methods_subspace: dict):
    # Load the data
    Xs, ys = load_test(f'{config["data_path"]}/{benchmark_set}/{ds_name}')

    # Execute
    results = []
    for i, (X, y) in enumerate(zip(Xs, ys)):

        for motif_discovery_name, motif_discovery_method in methods.items():
            motif_sets = motif_discovery_method(X, meta, **tuned_parameters[motif_discovery_name])
            results.append(pd.Series(
                {
                    'benchmark_set': benchmark_set,
                    'ds_name': ds_name,
                    'time_series_id': i,
                    'model': motif_discovery_name
                }
                |
                common.compute_scores(y, motif_sets, config['evaluation'])
            ))

        for motif_discovery_name, motif_discovery_method in methods_subspace.items():
            hyper_parameters, subspace_hyper_parameters = common.split_hyper_parameters(tuned_parameters[motif_discovery_name])
            univariate_motif_sets = motif_discovery_method(X, meta, **hyper_parameters)
            motif_sets = apply_sub_tsmd(univariate_motif_sets, **subspace_hyper_parameters)
            (results.append(pd.Series(
                {
                    'benchmark_set': benchmark_set,
                    'ds_name': ds_name,
                    'time_series_id': i,
                    'model': motif_discovery_name
                }
                |
                common.compute_scores(y, motif_sets, config['evaluation'])
            )))

    # Save the results
    path = f'{config["results_path"]}/test/{benchmark_set}'
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

    # Evaluate the models
    for ds_name, meta in metadata.iterrows():
        benchmark_set = benchmark_sets[ds_name]
        if ds_name == 'simple' and benchmark_set == 'synthetic':
            evaluate(config, tuned_parameters[benchmark_set]['s1'], benchmark_set, ds_name, meta, methods, methods_subspace)
        else:
            evaluate(config, tuned_parameters[benchmark_set][ds_name], benchmark_set, ds_name, meta, methods, methods_subspace)


if __name__ == '__main__':
    main(*common.parse_args(require_tuned_parameters=True))
