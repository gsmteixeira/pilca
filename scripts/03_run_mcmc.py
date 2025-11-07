import sys
sys.path.append("/tf/ProjectGabriel/pilca")

import numpy as np
import pandas as pd
from lightcurve_fitting import models, filters, lightcurve, fitting
import matplotlib.pyplot as plt
import matplotlib as mpl
import importlib
from utils.utils import load_lc
from utils.utils import light_curve_plot
import torch
import utils.torchphysics as tp
import utils.utils as ut
# import lc
import os
torch.set_default_dtype(torch.float64)

all_filters = ["z", "y", "i", "r", "g", "u", "uvw1"][::-1]  # from red to UV
mjd_array = np.linspace(3, 13, 600)


filter_combinations = [all_filters[:i+1] for i in range(len(all_filters))]

max_days = 5
time_spans = np.arange(1, max_days + 1)  # [1, 2, ..., 10]

model_parameters = [1.2, 2.,2., 4.0, 2.5]

# --- setup light curve builder ---
builder = ut.LCBuilder(
    model_name="sc4",
    model_parameters=model_parameters,
    model_units=[1,1,1,1,1],
    seed=42
)

lc = builder.build_sim_lc(
    mjd_array=mjd_array,
    filters_list=all_filters,  # full set
    redshift=0.00526,
    dlum_factor=1e-1/2,
    dm=31.1,
    dL=19.,
    dLerr=2.9
)

t_span = 10
min_mjd = lc['MJD'].min()

lc_early = lc.where(MJD_min=min_mjd-1, MJD_max=min_mjd+t_span) 

model = models.ModifiedShockCooling4(lc_early)


ten13cmtoRsol = 1e13*1.4374e-11 
ten8p5cmstoten3kms = 10**.5
# units_array = np.array([ten8p5cmstoten3kms, 1, 1, ten13cmtoRsol, 1, 1]).reshape(1,-1)

priors = [
models.UniformPrior(0, 10),
models.UniformPrior(0, 10),
models.UniformPrior(0, 10),
models.UniformPrior(0, 100),
models.UniformPrior(min_mjd-4, min_mjd+5),
models.LogUniformPrior(0,1e2),
]

p_lo = [0.5, 0.5,  0.5, 1, min_mjd-2, 6]
p_up = [3, 4,  4, 6, min_mjd+2, 8]

N_WALKERS = 40
N_STEPS = 8000
N_STEPS_BURNIN = 1 
sampler = fitting.lightcurve_mcmc(lc_early, model, 
                                priors=priors, p_lo=p_lo, p_up=p_up,
                                nwalkers=N_WALKERS, nsteps=N_STEPS, nsteps_burnin=N_STEPS_BURNIN, use_sigma=True)


save_dir = os.path.join(ut.storage_parent_path, "experiments", "mcmc")
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, "flatchain_mcmc_10d_all_filters.npy"), np.array(sampler.flatchain))

