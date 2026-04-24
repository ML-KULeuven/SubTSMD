
import os
import itertools

import toml
import pandas as pd

from sub_tsmd import load_validation, apply_sub_tsmd
import common


########################################################################
# UTILITY METHODS
########################################################################

def format_parameter_grid(parameter_grid: dict) -> list:
    if len(parameter_grid) == 0:
        return [{}]
    keys, values = zip(*parameter_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def save_results(config, results: list, benchmark_set: str, ds_name: str, motif_discovery_name: str):
    results_df = pd.concat(results, axis=1).T
    path = f'{config["results_path"]}/validation/{benchmark_set}/{ds_name}'
    if not os.path.exists(path):
        os.makedirs(path)
    results_df.to_csv(f'{path}/{motif_discovery_name}.csv', index=False)


########################################################################
# EVALUATION METHODS
########################################################################

def wrapper(config, X, y, ds_name, meta, motif_discovery_method, hyper_parameters):
    motif_sets = motif_discovery_method(X, meta, **hyper_parameters)
    return pd.Series({'ds_name': ds_name} | hyper_parameters | common.compute_scores(y, motif_sets, config['evaluation']))


def wrapper_subspace(config, X, y, ds_name, meta, motif_discovery_method, hyper_parameters):
    results = []
    univariate_motif_sets = motif_discovery_method(X, meta, **hyper_parameters)
    for subspace_hyper_parameters in format_parameter_grid(config['subspace']['parameter_grid']):
        motif_sets = apply_sub_tsmd(univariate_motif_sets, **subspace_hyper_parameters)
        results.append(pd.Series({'ds_name': ds_name} | hyper_parameters | subspace_hyper_parameters | common.compute_scores(y, motif_sets, config['evaluation'])))
    return pd.concat(results, axis=1)


def evaluate(config, benchmark_set, ds_name, meta, motif_discovery_name, motif_discovery_method, subspace: bool):
    # Gather the jobs
    Xs, ys = load_validation(f'{config["data_path"]}/{benchmark_set}/{ds_name}')
    jobs = [
        (config, X, y, ds_name, meta, motif_discovery_method, hyper_parameters)
        for X, y in zip(Xs, ys)
        for hyper_parameters in format_parameter_grid(config[motif_discovery_name]['parameter_grid'])
    ]

    print(f'Applying {"subspace" if subspace else ""} {motif_discovery_name} on {benchmark_set}/{ds_name}')
    print(f'Total number of jobs: {len(jobs)}')

    # Execute the jobs
    if subspace:
        results = [wrapper_subspace(*job) for job in jobs]
    else:
        results = [wrapper(*job) for job in jobs]

    # Save the results
    save_results(config, results, benchmark_set, ds_name, motif_discovery_name)


########################################################################
# MAIN LOOP
########################################################################

def main(config_path: str, datasets: list, methods: dict, methods_subspace: dict):
    """
    The ID's of the time series are not tracked with the current setup. This means that it
    is not possible to directly compare the methods on a specific time series. However, the
    goal of this experiment is to tune the models, for which the general trends suffice.

    Adding the ID is relatively straightforward: it should also be passed to the wrapper
    methods, which then also include it in the result frames. The tuning process should be
    re-executed then.
    """

    # Read the config file
    with open(config_path, 'r') as f:
        config = toml.load(f)

    # Load the metadata
    metadata, benchmark_sets = common.load_metadata(config["data_path"], datasets)

    # Evaluate the models
    for method_name, motif_discovery_method in methods.items():
        for ds_name, meta in metadata.iterrows():
            evaluate(config, benchmark_sets[ds_name], ds_name, meta, method_name, motif_discovery_method, subspace=False)

    # Evaluate the subspace models
    for method_name, motif_discovery_method in methods_subspace.items():
        for ds_name, meta in metadata.iterrows():
            evaluate(config, benchmark_sets[ds_name], ds_name, meta, method_name, motif_discovery_method, subspace=True)


if __name__ == '__main__':
    main(*common.parse_args(require_tuned_parameters=False))
