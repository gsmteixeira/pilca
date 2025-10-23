import sys
sys.path.append("/tf/ProjectGabriel/pilca")

import numpy as np
import pandas as pd
from lightcurve_fitting import models, filters, lightcurve
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


all_filters = ["z", "y", "i", "r", "g", "u", "uvw1"]  # from red to UV
filter_combinations = [all_filters[:i+1] for i in range(len(all_filters))]

max_days = 10
time_spans = np.arange(1, max_days + 1)  # [1, 2, ..., 10]

def run_experiment(filters, time_span, lc, true_params):
    """
    Run one experiment for given filter subset and time span (days).
    """
    # build time grid subset (from day 0 to time_span)
    mask_time = lc["MJD"] <= (lc["MJD"].min() + time_span)
    lc_subset = lc[mask_time & lc["filter"].isin(filters)]

    # prepare experiment folder
    logger = ExperimentLogger(base_dir="experiments")

    # torchenize
    torchenizer = tp.Torchenizer(lc_subset, ycol="loglum", device=device)
    X_DATA, filters_mask, ufilters = torchenizer.get_xdata(max_phase=time_span, t0_offset=3)

    # build and train model (same as before)
    nn_model = tp.MultiFilterBNN(x_data=X_DATA, filters_mask=filters_mask,
                                 param_dim=4, hidden_dim=32).to(device)
    criterion = tp.ModifiedSC4Loss(ufilters=ufilters,
                                   z=lc.meta["redshift"],
                                   mode="mean_param",
                                   min_t0=torch.min(X_DATA[:,0])-1e-5)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    trainer = tp.Trainer(model=nn_model, criterion=criterion, epochs=2000,
                         n_samples_loss=30, optimizer=optimizer,
                         scheduler=scheduler, verbose_step=200,
                         save_dir=os.path.join(logger.exp_dir, "model_weights.pth"))

    history = trainer.train()

    # sample posterior
    samples = torch.stack([nn_model() for _ in range(1000)]).detach().cpu().numpy()

    # save everything
    logger.save_model(nn_model)
    logger.save_history(history)
    logger.save_samples(samples)
    logger.save_config({
        "filters": filters,
        "time_span": time_span,
        "true_params": list(true_params),
        "n_points": len(lc_subset),
        "param_dim": 4,
        "epochs": 2000
    })
    logger.summarize(f"Filters={filters}, Time={time_span}d, Final loss={history['loss'][-1]:.4f}")


def main():

    builder = ut.LCBuilder(model_name="sc4",
                       model_parameters=[1.26491106, 2., 4.03506331, 2.5],
                       model_units=[1,1,1,1],
                       seed=42)

    lc = builder.build_sim_lc(mjd_array=np.linspace(3, 13, 300),
                            filters_list=["g", "r", "i"],
                            redshift=0.00526,
                            dlum_factor = 1e-1,
                            dm=31.1,
                            dL=19.,
                            dLerr=2.9)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torchenizer = tp.Torchenizer(lc, ycol="loglum", device=device)

    X_DATA, filters_mask, ufilters = torchenizer.get_xdata(max_phase=8, t0_offset=3)

    nn_model_name = "bnn_pilca.pth"
    model_save_dir = os.path.join(ut.storage_parent_path, "models", nn_model_name)

    nn_model = tp.MultiFilterBNN(x_data=X_DATA,
                                filters_mask=filters_mask,
                                param_dim=4,
                                hidden_dim=32).to(device)

    criterion = tp.ModifiedSC4Loss(ufilters=ufilters,
                                z=lc.meta["redshift"],
                                mode="mean_param",
                                min_t0=torch.min(X_DATA[:,0])-1e-5)
    lr=1e-3
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=500, gamma=0.5)

    trainer = tp.Trainer(model=nn_model,
                        criterion=criterion,
                        epochs=2000,
                        n_samples_loss=30,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        verbose_step=100,
                        save_dir=model_save_dir)

    history = trainer.train()

    samples = []
    for i in range(1000):
        samples.append(nn_model())
    samples = torch.stack(samples).detach().cpu().numpy() # shape [N_samples, 5]


    return

if __name__=="__main__":
    main()