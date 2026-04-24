# Running the experiments

This directory contains all the information required to execute and analyze the experiments. The ``analysis`` directory contains all the notebooks used for analyzing the results. The raw experimental results are in the ``results`` directory. The scripts used for running the experiments are available in ``scripts``. These are explained in more detail below. 

## Scripts

### ``validation.py``

The validation script can be run as follows:
```bash
python validation.py --config <path-to-config> --motif_discovery <motif-discovery-methods> --subspace_motif_discovery <subspace-motif-discovery-methods> --datasets <datasets>
```
With the following arguments:
- ``<path-to-config>``: The path towards the configuration (toml) file.
- ``<motif-discovery-methods>``: A list of motif discovery methods to use. Valid methods are 
  ``EMD_star``, ``LAMA`` and ``MMotifs``. It is also possible to pass either ``all`` or ``none``, in which case all the methods or none of the methods will be used, respectively. 
- ``<subspace-motif-discovery-methods>``: A list of motif discovery methods to use, which are combined with subspace motif discovery. Valid methods are ``GrammarVizRePair``, ``LatentMotifs``, ``LoCoMotif``, ``Motiflets`` and ``SetFinder``. It is also possible to pass either ``all`` or ``none``, in which case all the methods or none of the methods will be used, respectively. 
- ``<datasets>``: A list of datasets which should be used for motif discovery. The format of the datasets equals `<benchmark_set>/<ds_name>``. If you set ``<ds_name>`` as ``*`` (e.g., ``synthetic/*``), then all the datasets from that benchmark set will be used. 

This script will evaluate all the given models on all the given datasets and write away the results in csv-files. The files are stored according to the following format: ``<benchmark_set>/<ds_name>/<method>.csv`` with ``<benchmark_set>`` the name of the benchmark (e.g., ``synthetic``), ``<ds_name>`` the name of the dataset within the benchmark (e.g., ``base``), and ``<method>`` is the name of the motif discovery method (e.g., ``LoCoMotif``).

### ``test.py``

The test script can be run as follows:
```bash
python test.py --config <path-to-config> --tuned_parameters <path-to-tuned-parameters> --motif_discovery <motif-discovery-methods> --subspace_motif_discovery <subspace-motif-discovery-methods> --datasets <datasets>
```
The arguments are identical as for ``validation.py``, except for:
- ``<path-to-tuned-parameters>``: The path to the tuned hyperparameters. This is the file as created in the ``analysis/validation.ipynb`` notebook. 

The script will evaluate each model on each dataset using the hyperparameters tuned for that dataset. For each dataset ``<ds_name>`` in benchmark set ``<benchmark_set>``, the script will create a csv-file ``<benchmark_set>/<ds_name>.csv``, in which each row gives the performance of a specific model on a specific time series in the dataset. 

### ``scalability.py``

The scalability script can be run as follows:
```bash
python scalability.py --config <path-to-config> --tuned_parameters <path-to-tuned-parameters> --motif_discovery <motif-discovery-methods> --subspace_motif_discovery <subspace-motif-discovery-methods> --datasets <datasets>
```
The arguments are identical as for ``test.py``. The output of the scalability script is also similar as for the test script, but instead of having the performance metrics in each row, it contains the total running time. For subspace methods, there are two additional columns, one containing the time for univariate motif discovery and one containing the time for merging the motifs. 

### ``scalability-SubTSMD.py``

This script runs scalability experiments specifically for SubTSMD. In these experiments we isolate the runtime of SubTSMD by simply generating motif sets (instead of time series) containing motifs. Because SubTSMD is extremely fast compared to motif discovery itself, we
can scale-up the size of the experiments to get a better understanding of its scalability. The script can be run as follows:
```bash 
python .\scalability-SubTSMD.py --config <path-to-config>
```
The ``<path-to-config>`` is identical as in the previous scripts.

### ``motif-pairs.py``

This script executes the experiments regarding motif-pair discovery, and can be executed as follows:
```bash
python motif-pairs.py --config <path-to-config> --motif_discovery <motif-discovery-methods> --subspace_motif_discovery <subspace-motif-discovery-methods>
```
The arguments are identical as for ``test.py``, with two minor modifications:
1. There is no need to provide any datasets. The script will automatically search for the motif-pair datasets (within the data-path in the config file), and execute the given motif discovery methods time series.
2. The only valid subspace motif discovery method in ``subspace_motif_discovery`` is ``'locomotif'``, along with the predefined keywords ``'all'`` and ``'none'``. 
