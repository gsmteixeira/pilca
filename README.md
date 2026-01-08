## PILCA — Physics-Informed Machine Learning Light Curve Analyzer

PILCA is a physics-informed machine-learning framework for the characterization of astrophysical
transients from sparse and irregular multi-band light curves. It combines differentiable physical
models with neural-network inference to provide fast, physically motivated parameter estimates and
reliable initial conditions for Bayesian methods such as MCMC.


**WORK IN PROGRESS — This project is in its very initial stages and under active development.**

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
```text
.
├── data/                   Observed Light Curves
├── lightcurve_fitting/     Clone of the [lightcurve_fitting](https://github.com/griffin-h/lightcurve_fitting) package from Griffin Hosseinzadeh
├── notebooks/              Main experiments and exploratory notebooks
├── scripts/                Routines and MCMC execution scripts
├── utils/                  Shared utilities and helpers
└── requirements.txt        Python dependencies
```

---------------------------------------------------------------------

MAIN COMPONENTS

data/
    The obseved photometry of a given SN. 

lightcurve_fitting/
        Original base code for the shock cooling models fitting (MCMC implementation) from Hosseinzadeh

notebooks/
    Pipeline building, test experiments, results and main workflow

scripts/
    Long experiments runner and MCMC execution

utils/
    Analysis, plotting, and differentiable-physics helpers. The file
    torchphysics.py contains the core modules of this project -- the Bayesian networks and the addapted shock cooling models.

---------------------------------------------------------------------

REPRODUCIBILITY AND DOCUMENTATION

This repository supports reproducibility of the PILCA methodology and its validation on simulated or observational data. Code documentation was generated with the assistance of LLMs.

For questions or contributions, please open an issue.
