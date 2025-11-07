import sys
sys.path.append("/tf/ProjectGabriel/pilca")

import os
import json
import datetime
import numpy as np
import torch
import pandas as pd

import utils.torchphysics as tp
import utils.utils as ut 
import torch
from lightcurve_fitting import filters as flc
import time

torch.set_default_dtype(torch.float64)




class ExperimentLogger:
    def __init__(self, base_dir="experiments", filters=None, time_span=None):
        os.makedirs(base_dir, exist_ok=True)
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = ""
        if filters is not None:
            suffix += f"_filters-{'-'.join(filters)}"
        if time_span is not None:
            suffix += f"_time-{time_span}d"
        # name = "exp_" if filters else "" 
        if filters:
            self.exp_dir = os.path.join(base_dir, f"exp_{suffix}")
        else:
            self.exp_dir = base_dir
        
        os.makedirs(self.exp_dir, exist_ok=True)

    def save_config(self, config, name=None):
        if not name:
            name = "exp_config.json"
        with open(os.path.join(self.exp_dir, name), "w") as f:
            json.dump(config, f, indent=4)

    def save_model(self, model):
        torch.save(model.state_dict(), os.path.join(self.exp_dir, "model_weights.pth"))

    def save_history(self, history):
        np.save(os.path.join(self.exp_dir, "loss_history.npy"), history["loss"])

    def save_samples(self, samples):
        np.save(os.path.join(self.exp_dir, "samples.npy"), samples)
    
    def save_history(self, history):
        """
        Save training history in both .npy and .csv formats.
        """
        import pandas as pd
        import numpy as np
        import os

        losses = np.array(history["loss"])

        # --- Save .npy ---
        # np.save(os.path.join(self.exp_dir, "loss_history.npy"), losses)

        # --- Save .csv ---
        df = pd.DataFrame(history)
        csv_path = os.path.join(self.exp_dir, "history.csv")
        df.to_csv(csv_path, index=False)


    def summarize(self, text):
        with open(os.path.join(self.exp_dir, "summary.txt"), "w") as f:
            f.write(text)




def get_filter_mask(lc, filters):
    filter_objects = [flc.filtdict[f] for f in filters]
    fmask = np.in1d(lc["filter"].value, filter_objects)
    return fmask

def run_experiment(lc, filters, time_span,
                   device, exp_base_dir, hyper_config):
    """
    Run one experiment for given filter subset and time span (days).
    """
    # build time grid subset (from day 0 to time_span)
    # mask_time = lc["MJD"] <= (lc["MJD"].min() + time_span)
    lc_subset = lc.where(MJD_max=lc["MJD"].min() + time_span)
    lc_subset = lc_subset[get_filter_mask(lc_subset, filters)]
    #[mask_time & lc["filter"].isin(filters)]

    # hyper config
    hc = hyper_config

    # prepare experiment folder
    logger = ExperimentLogger(base_dir=exp_base_dir, filters=filters,
                              time_span=time_span)

    # torchenize
    torchenizer = tp.Torchenizer(lc_subset, ycol="loglum", device=device)
    X_DATA, filters_mask, ufilters = torchenizer.get_xdata(max_phase=time_span, 
                                                           t0_offset=hc["data"]["t0_offset"])

    # build and train model (same as before)
    nn_model = tp.MultiFilterBNN(x_data=X_DATA, filters_mask=filters_mask,
                                 param_dim=hc["model"]["param_dim"], 
                                 hidden_dim=hc["model"]["hidden_dim"],
                                 n_filter_layers=hc["model"]["n_filter_layers"],
                                 n_combined_layers=hc["model"]["n_combined_layers"]
                                 ).to(device)
    criterion = tp.ModifiedSC4Loss(ufilters=ufilters,
                                   z=lc.meta["redshift"],
                                   mode=hc["training"]["loss_mode"],
                                   min_t0=torch.min(X_DATA[:,0])-1e-5)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=hc["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=hc["scheduler"]["step_size"],
                                                 gamma=hc["scheduler"]["gamma"])

    trainer = tp.Trainer(model=nn_model,
                         criterion=criterion, epochs=hc["training"]["epochs"],                         
                         n_samples_loss=hc["training"]["n_samples_loss"], 
                         optimizer=optimizer,
                         scheduler=scheduler, 
                         verbose_step=100,
                         save_dir=os.path.join(logger.exp_dir, "model_weights.pth"),
                         es_kwargs=hc["early_stop"],
                         change_loss=hc["training"]["change_loss"])

    history = trainer.train()

    # sample posterior
    samples = torch.stack([nn_model() for _ in range(hc["sampling"]["n_samples_save"])]).detach().cpu().numpy()

    # save everything
    logger.save_model(nn_model)
    logger.save_history(history)
    logger.save_samples(samples)
    logger.save_config({
        "filters": filters,
        "time_span": time_span+0.,
        "n_points": len(lc_subset)+0.,
    })
    logger.summarize(f"Filters={filters}, Time={time_span}d, Final loss={history['loss'][-1]:.4f}")



def main():

    IS_A_TEST = True
    TEST_NAME = "TEST_DEBUG_1_free_frhom"

    # --- define cumulative filter sets (z → UV) ---

    all_filters = ["z", "y", "i", "r", "g", "u", "uvw1"][::-1]  # from UV to red
    mjd_array = np.linspace(3, 13, 600)


    filter_combinations = [all_filters[:i+1] for i in range(len(all_filters))]

    max_days = 10
    time_spans = np.arange(1, max_days + 1)  # [1, 2, ..., 10]

    model_parameters = [1.2, 2., 2., 4.0, 2.5]

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
                            dlum_factor=1e-1/2 ,
                            dm=31.1,
                            dL=19.,
                            dLerr=2.9
                        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    param_dir_name = "-".join([f"{str(v).replace('.', 'p')}" for v in model_parameters])

    if IS_A_TEST:
        exp_base_dir = os.path.join(ut.storage_parent_path, "experiments", TEST_NAME)

    else:
        exp_base_dir = os.path.join(ut.storage_parent_path, "experiments", param_dir_name)

    hyper_log = ExperimentLogger(exp_base_dir)
    hyper_config = {
                    "learning_rate": 1e-2,
                    "scheduler": {
                        "step_size": 150,
                        "gamma": 0.5
                    },
                    "training": {
                        "epochs": 2000,
                        "n_samples_loss": 50,
                        "loss_mode": "mean_param",
                        "change_loss":True
                    }, 
                    "sampling": {
                        "n_samples_save": 1000
                    },
                    "model": {
                        "hidden_dim": 16,
                        "n_filter_layers":1,
                        "n_combined_layers":2,
                        "param_dim": 5
                    },
                    "data": {
                        "t0_offset": 3
                    },
                    "early_stop": {
                        "use_es":True,
                        "patience":200,
                        "save_best_loss":True}
                }
    
    hyper_log.save_config(hyper_config, name="hyper_config.json")

    # --- run experiments ---
    for filt_subset in filter_combinations:
        for tspan in time_spans:
            print(f"\n Running experiment: filters={filt_subset}, time_span={tspan} days")
            try:

                run_experiment(lc, filt_subset, tspan,
                           device, exp_base_dir,
                           hyper_config)
                time.sleep(30)
            except:
                raise Warning(f"Failed in experiment \n filters-{'-'.join(filt_subset)}; t={tspan}d\n")
                # aa
                continue

if __name__=="__main__":
    main()
