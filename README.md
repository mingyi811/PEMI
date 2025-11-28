# PEMI

This repository contains the code to reproduce the results in the paper [Online Selective Conformal Prediction with Asymmetric Rules:
A Permutation Test Approach](https://ying531.github.io/assets/files/PEMI_paper.pdf).

## Files
- `application/`: Real-data applications on drug discovery.
    - `application/cov_dep/`: Covariate-dependent selection.
      - `application/cov_dep/dec/`: Decision-driven selection.
      - `application/cov_dep/wq_wa/`: Selection based on weighted quantile or average.
      - `application/cov_dep/opt/`: Selection based on model uncertainty.
    - `application/conf_sel/`: Conformal selection.
    - `application/earlier_out/`: Selection based on earlier outcomes.
- `simulation/`: Simulation experiments on various settings.
- `data/`: Drug discovery datasets used in application experiments.
- `requirements.txt`: A list of Python packages of my environment.

## 1. Application

The application experiments are organized into separate folders, each corresponding to a particular selection rule.  
Within each folder, the following files are provided:

- `config.py` — Defines hyperparameters and experimental settings.  
- `data.py` — Loads and preprocesses the drug discovery dataset.  
- `interval.py` — Constructs the prediction sets used in the experiments.  
- `parallel_experiment.py` — Runs the full experimental pipeline, including executing experiments, recording results, and generating preliminary plots.  
- `selection.py` — Implements the selection rules associated with the corresponding folder.

To reproduce our results, users may set hyperparameters directly in `config.py` and then run:

```
python parallel_experiment.py
```

## 2. Simulation

The simulation code follow the same structure as the application code. 

The main differences from the application setting are:

- the **data-generating process** in `data.py` has been replaced with two synthetic models;  
- `interval.py` includes an additional option for the **CQR score function**.

To reproduce any simulation experiment, one can adjust the settings and code in each file, and run:

```
python parallel_experiment.py
```




