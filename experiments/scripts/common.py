
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

from competitors import *
from sub_tsmd import (
    matching_matrix,
    micro_averaged_precision,
    micro_averaged_recall,
    micro_averaged_f1,
    macro_averaged_precision,
    macro_averaged_recall,
    macro_averaged_f1,
)


def parse_args(*, require_tuned_parameters=False):

    # All models
    motif_discovery_methods = {
        'EMD_star': emd_star,
        'LAMA': lama,
        'MMotifs': mmotifs,
    }
    subspace_motif_discovery_methods = {
        'GrammarVizRePair': univariate_grammar_viz_repair,
        'LatentMotifs': univariate_latent_motifs,
        'LoCoMotif': univariate_locomotif,
        'Motiflets': univariate_motiflets,
        'SetFinder': univariate_set_finder,
    }

    # Setup the args
    parser = argparse.ArgumentParser(description="Script to run models on datasets with a configuration file.")
    parser.add_argument('--config', type=str, required=True, help=f'Path to the configuration file')
    if require_tuned_parameters:
        parser.add_argument('--tuned_parameters', type=str, required=True, help=f'Path to the configuration file with tuned hyperparameters')
    parser.add_argument('--motif_discovery', type=str, nargs='+', required=True, help=f'List of models to use.\nValid options: {list(motif_discovery_methods.keys())}')
    parser.add_argument('--subspace_motif_discovery', type=str, nargs='+', required=True, help=f'List of models to use.\nValid options: {list(subspace_motif_discovery_methods.keys())}')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of datasets to use')
    args = parser.parse_args()

    # Extract the selected motif discovery methods
    if args.motif_discovery == ['all']:
        selected_motif_discovery_methods = motif_discovery_methods.keys()
    elif args.motif_discovery == ['none']:
        selected_motif_discovery_methods = []
    else:
        selected_motif_discovery_methods = args.motif_discovery

    # Extract the selected subspace motif discovery methods
    if args.subspace_motif_discovery == ['all']:
        selected_subspace_motif_discovery_methods = subspace_motif_discovery_methods.keys()
    elif args.subspace_motif_discovery == ['none']:
        selected_subspace_motif_discovery_methods = []
    else:
        selected_subspace_motif_discovery_methods = args.subspace_motif_discovery

    # Return the data
    if require_tuned_parameters:
        return (
            args.config,
            args.tuned_parameters,
            args.datasets,
            {k: motif_discovery_methods[k] for k in selected_motif_discovery_methods},
            {k: subspace_motif_discovery_methods[k] for k in selected_subspace_motif_discovery_methods}
        )
    else:
        return (
            args.config,
            args.datasets,
            {k: motif_discovery_methods[k] for k in selected_motif_discovery_methods},
            {k: subspace_motif_discovery_methods[k] for k in selected_subspace_motif_discovery_methods}
        )


def load_metadata(data_path: str, datasets: list):
    # Configure the metadata
    metadata = []
    benchmark_sets = {}
    for dataset in datasets:
        benchmark_set, ds_name = dataset.split('/')
        benchmark_metadata = pd.read_csv(f'{data_path}/{benchmark_set}/metadata.csv')
        if ds_name == '*':
            metadata.append(benchmark_metadata)
            for name in benchmark_metadata['ds_name']:
                benchmark_sets[name] = benchmark_set
        else:
            metadata.append(pd.DataFrame(benchmark_metadata.set_index('ds_name').loc[ds_name]).T.reset_index(names='ds_name'))
            benchmark_sets[ds_name] = benchmark_set
    return pd.concat(metadata).drop_duplicates().set_index('ds_name'), benchmark_sets


def compute_scores(y_true: List[Tuple[np.array, np.ndarray]], y_pred: List[Tuple[np.array, np.ndarray]], config: dict) -> dict:
    scores = {}
    for t in config['thresholds_subspace']:
        M, _, _ = matching_matrix(y_true, y_pred, threshold_subspace=t, threshold_ovr=config['threshold_ovr'])

        scores[f'P-micro@{t}'] = micro_averaged_precision(M)
        scores[f'R-micro@{t}'] = micro_averaged_recall(M)
        scores[f'F1-micro@{t}'] = micro_averaged_f1(M)
        scores[f'P-macro@{t}'] = macro_averaged_precision(M)
        scores[f'R-macro@{t}'] = macro_averaged_recall(M)
        scores[f'F1-macro@{t}'] = macro_averaged_f1(M)

    # Convert NAN-values to 0
    for k, v in scores.items():
        if np.isnan(v):
            scores[k] = 0.0

    return scores


def split_hyper_parameters(hyper_parameters: dict) -> (dict, dict):
    remaining_hyper_parameters = hyper_parameters.copy()
    subspace_hyper_parameters = {k: remaining_hyper_parameters.pop(k) for k in ['linkage', 'delta']}
    return remaining_hyper_parameters, subspace_hyper_parameters


########################################################################
# MOTIF DISCOVERY
########################################################################

def window_size(l_min: int, l_max: int) -> int:
    return int((l_min + l_max) / 2)


def k_max(meta: pd.Series) -> int:
    if meta.name in [f'e{i+1}' for i in range(8)]:
        return 20
    else:
        return 10


def emd_star(X, meta, n_dims, elbow_deviation):
    return emd_star_wrapper(
        X,
        l_min=int(meta['l_min']),
        l_max=int(meta['l_max']),
        k_max=k_max(meta),
        n_dims=n_dims,
        elbow_deviation=elbow_deviation
    )


def lama(X, meta, n_dims, elbow_deviation):
    return lama_wrapper(
        X,
        l_min=int(meta['l_min']),
        l_max=int(meta['l_max']),
        k_max=k_max(meta),
        n_dims=n_dims,
        elbow_deviation=elbow_deviation
    )


def mmotifs(X, meta):
    return mmotifs_wrapper(
        X,
        window_size=window_size(int(meta['l_min']), l_max=int(meta['l_max'])),
        r=meta['r'],
        max_motifs=5
    )


def univariate_grammar_viz_repair(X, meta, alphabet_size, word_size):
    raise ValueError('Implementation for grammar viz is not publicly available.')
    # return univariate_gv_repair_wrapper(
    #     X,
    #     window_size=window_size(int(meta['l_min']), l_max=int(meta['l_max'])),
    #     alphabet_size=alphabet_size,
    #     word_size=word_size
    # )


def univariate_latent_motifs(X, meta):
    raise ValueError('Implementation for latent motifs is not publicly available.')
    # return univariate_latent_motif_wrapper(
    #     X,
    #     window_size=window_size(int(meta['l_min']), l_max=int(meta['l_max'])),
    #     r=meta['r']
    # )


def univariate_locomotif(X, meta, rho, warping):
    return univariate_locomotif_wrapper(
        X,
        l_min=int(meta['l_min']),
        l_max=int(meta['l_max']),
        rho=rho,
        warping=warping
    )


def univariate_motiflets(X, meta, elbow_deviation):
    raise ValueError('Implementation for motiflets is not publicly available.')
    # return univariate_motiflets_wrapper(
    #     X,
    #     l_min=int(meta['l_min']),
    #     l_max=int(meta['l_max']),
    #     k_max=k_max(meta),
    #     elbow_deviation=elbow_deviation
    # )


def univariate_set_finder(X, meta):
    raise ValueError('Implementation for set finder is not publicly available.')
    # return univariate_set_finder_wrapper(
    #     X,
    #     window_size=window_size(int(meta['l_min']), l_max=int(meta['l_max'])),
    #     r=meta['r']
    # )
