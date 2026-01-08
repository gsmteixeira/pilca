PILCA — Physics-Informed Machine Learning Light Curve Analyzer

PILCA is a physics-informed machine-learning framework for the characterization of astrophysical
transients from sparse and irregular multi-band light curves. It combines differentiable physical
models with neural-network inference to provide fast, physically motivated parameter estimates and
reliable initial conditions for Bayesian methods such as MCMC.

The framework is developed with relevance to large time-domain surveys (e.g. LSST), where early,
automated physical interpretation of transient alerts is essential.

---------------------------------------------------------------------

SCOPE

This repository focuses on:
- Physics-informed neural inference for transient light curves
- Differentiable forward modeling of physical emission models
- Data-efficient parameter estimation under sparse cadence
- Preparation of initial conditions for Bayesian inference

This is a methodological and scientific framework, not a full alert-broker system.

---------------------------------------------------------------------

REPOSITORY STRUCTURE

.
├── data/                   Example or intermediate data
├── lightcurve_fitting/     Core PILCA package
├── notebooks/              Tutorials and exploratory notebooks
├── scripts/                Experiment and MCMC execution scripts
├── utils/                  Shared utilities and helpers
└── requirements.txt        Python dependencies

---------------------------------------------------------------------

MAIN COMPONENTS

lightcurve_fitting/
    Neural networks, differentiable physical models, and training logic

notebooks/
    End-to-end examples (training, evaluation, MCMC preparation)

scripts/
    Reproducible experiment runners

utils/
    Analysis, plotting, and differentiable-physics helpers

---------------------------------------------------------------------

DATA

Raw survey data are not included. The code assumes user-provided or simulated multi-band light
curves. Data paths must be configured locally in notebooks or scripts.

---------------------------------------------------------------------

REPRODUCIBILITY AND DOCUMENTATION

This repository supports reproducibility of the PILCA methodology and its validation on simulated
or observational data. Code documentation (docstrings and comments) was generated with the
assistance of large language models and manually reviewed.

For questions or contributions, please open an issue.
