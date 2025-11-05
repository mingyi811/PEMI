# PEMI

This repository contains the code to reproduce the results in the paper

## Files
- `application/`: Real-data applications on drug discovery.
    - `application/covariate-dependent selection rule/`: Drug discovery with covariate-dependent selection rules.
      - `application/covariate-dependent selection rule/decision-driven selection rule/`: Drug discovery with decision-driven selection rules.
      - `application/covariate-dependent selection rule/weighted quantile or average/`: Drug discovery with selection rules based on weighted quantile or average.
      - `application/covariate-dependent selection rule/online optimization selection rule/`: Drug discovery with selection rules based on model uncertainty.
    - `application/conformal p-value selection rule/`: Drug discovery with conformal selection including p-value and e-value.
    - `application/earlier outcomes selection rule/`: Drug discovery with selection rules based on earlier outcomes.
- `simulation/`: Simulation experiments on various settings.
- `data/`: Drug discovery datasets used in application experiments.
- `requirements.txt`: A list of Python packages of my environment(including some unnecessary packages).

## 1. Application

For application experiments, I have organized a folder for each selection rule to facilitate reproducibility.
Each folder contains the following files:

- `config.py`: Defines various hyperparameters used in the experiments.  
- `data.py`: Loads and preprocesses the dataset.  
- `interval.py`: Constructs the prediction sets.  
- `parallel_experiment.py`: Runs the entire experimental pipeline — including executing experiments, recording results, and generating preliminary plots. One can directly run this file to reproduce the experiments.  
- `selection.py`: Implements the selection rules used in the experiments.

## 2. Simulation

In the simulation part, we only varied the experimental settings to demonstrate the performance of our method under different scenarios. Therefore, there is no essential change in the code structure.

The file structure for simulation experiments is identical to that of the `weighted quantile or average` selection rule in the application section.  The only differences are that the data-generating process has been modified to accommodate two synthetic data settings, and an additional option for the CQR score function has been incorporated into `interval.py`.  

One can adjust the settings and code in each file to fully reproduce our simulation experiments.



