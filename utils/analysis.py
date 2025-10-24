import sys
sys.path.append("/tf/ProjectGabriel/pilca")
import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.table import Table
from pkg_resources import resource_filename
from abc import ABCMeta, abstractmethod
from scipy.interpolate import CubicSpline
import torch
from zmq import device
import torch
# from .filters import filtdict
from extinction import fitzpatrick99
import torch.nn as nn
import torch.nn.functional as F
from lightcurve_fitting import filters as flc
from lightcurve_fitting import models as mlc
from scipy.stats import gaussian_kde


def get_filter_mask(lc, filters):
    filter_objects = [flc.filtdict[f] for f in filters]
    fmask = np.in1d(lc["filter"].value, filter_objects)
    return fmask


def get_perdev(true, hat):
    return (true - hat) / true

def get_chi2(y_true, y_fit, errors=None):
    if errors is None:
        errors = np.ones_like(y_true)
    chi2 = np.sum(((y_true - y_fit) / errors) ** 2)
    return chi2 / len(y_true)

def get_logprob(true, samples):
    # mu = np.mean(samples, axis=0)
    # print(mu)
    # kde = gaussian_kde(samples.T)
    # x = np.array(true)
    # p = kde.evaluate(x)[0]

    # Compute empirical CDF for each parameter
    cdf_values = []

    for i in range(samples.shape[1]):
        sorted_samples = np.sort(samples[:, i])
        cdf = np.searchsorted(sorted_samples, true[i], side="right") / len(sorted_samples)
        cdf_values.append(cdf)

    cdf_values = np.array(cdf_values)

    # cov = np.cov(samples, rowvar=False)
    # diff = true - mu
    # inv_cov = np.linalg.pinv(cov)
    # log_det = np.log(np.linalg.det(cov) + 1e-12)
    # logprob = -0.5 * (np.dot(diff, np.dot(inv_cov, diff)) + log_det + len(true) * np.log(2 * np.pi))
    # prob = np.exp(logprob)
    return cdf_values

def get_loglum(model, params, lc=None, f=None):
    # assumes model returns luminosity; we take log10
    lum = model(t_in =lc["MJD"], v_s=params[0],M_env=params[1],f_rho_M=params[1],R=params[2],t_exp=params[3], f=lc["filter"])
    return np.log10(np.maximum(lum, 1e-12))

def evaluate_sampling(lc, model_name, samples, true_p):

    if model_name=="sc4":
        model = mlc.ShockCooling4(lc)
    else:
        raise Exception("no model provided")

    y_fit = np.zeros(len(lc))
    ufilters = np.unique(lc["filter"])
    # fmasks = [get_filter_mask(lc, uf) for uf in ufilters]

    sampling_p = samples.mean(0)

    # for i in range(len(ufilters)):
    #     y_f = get_loglum(model, sampling_p, lc, f=ufilters[i])
    #     y_fit[fmasks[i]] = y_f
    
    y_fit = get_loglum(model, sampling_p, lc)

    y_true = lc["loglum"].value

    # print(lc["dloglum"].value)
    # aaa

    chi2 = get_chi2(y_true, y_fit, )#errors=lc["dloglum"].value)
    cdf = get_logprob(true_p, samples)
    perdev = get_perdev(true_p, sampling_p)

    result = {"chi2": chi2,
              "cdf": cdf,
              "perdev": perdev}
    return result

# def evaluate_sampling(lc, samples, true_params, model_func, errors=None):
#     """Compute mean params, chi2, and per-parameter MSE."""
#     mean_params = samples.mean(dim=0)

#     model_mean = model_func(mean_params)
#     model_true = model_func(true_params)

#     if errors is None:
#         errors = torch.ones_like(data)

#     chi2 = torch.sum(((model_mean - model_true) / errors) ** 2).item()
#     mse_params = ((mean_params - true_params) ** 2).detach().cpu().numpy()

#     param_names = [f"param_{i}" for i in range(len(true_params))]
#     summary_df = pd.DataFrame({
#         "Parameter": param_names,
#         "True": true_params.cpu().numpy(),
#         "Mean": mean_params.detach().cpu().numpy(),
#         "MSE": mse_params
#     })
#     summary_df.loc[len(summary_df.index)] = ["χ² (model)", np.nan, np.nan, chi2]
#     return summary_df, model_mean, model_true
