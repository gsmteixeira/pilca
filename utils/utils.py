import sys
sys.path.append("/tf/ProjectGabriel/pilca")

import numpy as np
import pandas as pd
from lightcurve_fitting import models, filters, lightcurve
import matplotlib.pyplot as plt
import matplotlib as mpl
import importlib
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import utils.torchphysics as tp


storage_parent_path = "/tf/astrodados2/pilca.storage/"
        




class LCBuilder():
    def __init__(self, model_name="sc4",
                  model_parameters=[1.26491106, 2., 4.03506331, 2.5],
                  model_units=[1,1,1,1],):

        self.model_name = model_name
        self.model_parameters = model_parameters
        self.model_units = model_units
        self.model_inputs = np.array(self.model_parameters)*np.array(self.model_units)


    def build_sim_lc(self, mjd_array=np.linspace(3, 13, 300),
                    filters_list=["g", "r", "i"],
                    redshift=0.00526,
                    dlum_factor = 1e-1,
                    dm=31.1,
                    dL=19.,
                    dLerr=2.9):
        
        lc_fake = load_lc(fake=True)
        lc_fake.meta['redshift'] = redshift
        lc_fake.meta['dm'] = dm
        model_inputs = np.array(self.model_parameters)*np.array(self.model_units)

        filters_list_obj = [filters.filtdict[f] for f in filters_list]
        filter_for_mjd = np.random.choice(filters_list_obj, size=len(mjd_array), replace=True)

        if self.model_name=="sc4":
            model = models.ShockCooling4(lc_fake)   
            v_s = model_inputs[0]       # Shock velocity
            M_env = model_inputs[1]     # Envelope mass
            f_rho_M = model_inputs[1]   # Density profile factor
            R = model_inputs[2]         # Radius
            t_exp = model_inputs[3]     # explosion time
            lum = model(mjd_array, v_s=v_s, M_env=M_env, f_rho_M=f_rho_M, R=R, t_exp=t_exp, f=filter_for_mjd)
            dlum = np.random.normal(0, dlum_factor*np.mean(lum), len(mjd_array))
        else:
            ValueError("Model doesn't exist")

        # flux = lum / (4 * np.pi * dL**2)
        # dflux = 2*flux*dLerr/dL
        loglum = np.log10(lum+dlum)
        dloglum = 1/np.log(10)*dlum/lum

        lc = load_lc(lc=lightcurve.LC({"MJD":mjd_array,
                    "lum":lum+dlum,#*np.random.choice([1.,-1], size=len(dlum)),
                    "dlum":np.random.choice(np.abs(dlum), size=len(dlum)),
                    "loglum":loglum,
                    "dloglum":np.random.choice(np.abs(dloglum), size=len(dloglum)),
                    "filter":filter_for_mjd}),
                    fake=False)
        
        lc.meta['redshift'] = redshift
        lc.meta['dm'] = dm
        # lc.calcLum()
        # lc.calcAbsMag()
        # lc.calcMag()

        return lc 
    


# def 
def load_lc(lc=None,
            fake=True,
            lc_dir=None,
            mw_extinction_dict={
            'U_S': 0.125,
            'B_S':0.103,
            'V_S':0.082,
            'U': 0.131,
            'B': 0.109,
            'V': 0.083,
            'R': 0.065,
            'I': 0.045,
            'u': 0.128,
            'g': 0.100,
            'c': 0.085,
            'r': 0.069,
            'o': 0.060,
            'i': 0.051,
            'z': 0.038,
            'UVW2': 0.234,
            'UVM2': 0.211,
            'UVW1': 0.166,
            'DLT40': 0.083,},
            host_extinction_dict={
            'U': 0.,
            'B': 0.,
            'V': 0.,
            'R': 0.,
            'I': 0.,
            'u': 0.,
            'g': 0.,
            'r': 0.,
            'i': 0.,
            'z': 0.,
            'UVW2': 0.,
            'UVM2': 0.,
            'UVW1': 0.,
            'o': 0.,
            'c': 0.,
            'U_S': 0.,
            'B_S': 0.,
            'V_S': 0.,
            'DLT40': 0.,
            },
            dm=31.39,
            z=0.00526):
    if fake:
        lc =  lightcurve.LC({"MJD":[2,3,4],
                "mag":[16,17,18],
                "filters":[filters.filtdict["g"],filters.filtdict["g"],filters.filtdict["g"]]})
    if lc_dir:
        lc = lightcurve.LC.read(lc_dir)

    lc.meta['dm'] = dm
    lc.meta['extinction'] = mw_extinction_dict
    # Based on no Na I D detection in spectrum
    lc.meta['hostext'] = host_extinction_dict

    # z = 0.00526
    lc.meta['redshift'] = z  # redshift
    return lc


def light_curve_plot(lc, offset=0.5, ycol="lum"):
    """
    Plots light curves with different markers, applying an offset for each filter.
    
    Parameters:
    lc : dict or structured array
        A dataset containing 'MJD', 'mag', 'dmag', and 'filter' fields.
    """

    # Unique filters in the dataset
    ufilts = np.unique(lc['filter'])

    # Define marker styles for each filter
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
    face_color = []
    
    # Define offsets for each filter (spaced by 0.5 mag)
    offsets = {filt: -10 + i * offset for i, filt in enumerate(ufilts[::-1])}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)  # Set figure size

    for i, filt in enumerate(ufilts):
        # Create mask for the current filter
        fmask = np.array(lc['filter'] == filt)
        
        # Extract values and apply offset
        mjd_filt = lc['MJD'][fmask]
        y_filt = lc[ycol][fmask] + offsets[filt]  # Apply offset
        y_filt_err = lc["d"+ycol][fmask]
        
        # Select marker style based on index
        marker = markers[i % len(markers)]
        
        style = filt.plotstyle

        # Plot with error bars
        ax.errorbar(mjd_filt, y_filt, yerr=y_filt_err,
         fmt=marker, label=f"{filt} (offset: {offsets[filt]:.1g})",
         capsize=3, **style)

    # Aesthetics
    ax.set_xlabel("MJD")
    ax.set_ylabel(f"{ycol} + Offset")
    # ax.invert_yaxis()  # Invert y-axis for magnitudes
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Light Curve Plot with Offsets")
    if ycol=="lum":
        ax.set_yscale("log")
    # Move the legend to the right of the plot
    ax.legend(title="Filter (Offset)", loc='center left', bbox_to_anchor=(1, 0.5))
    


def pilca_light_curve_plot(lc, offset=0.5, pilcas=None, ufilters=None):
    """
    Plots light curves with different markers, applying an offset for each filter.
    
    Parameters:
    lc : dict or structured array
        A dataset containing 'MJD', 'mag', 'dmag', and 'filter' fields.
    """

    # Unique filters in the dataset
    ufilts = ufilters#np.unique(lc['filter'])

    # Define marker styles for each filter
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
    face_color = []
    
    # Define offsets for each filter (spaced by 0.5 mag)
    offsets = {filt: -10 + i * offset for i, filt in enumerate(ufilts[::-1])}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)  # Set figure size

    for i, filt in enumerate(ufilts):
        # Create mask for the current filter
        fmask = np.array(lc['filter'] == filt)
        
        # Extract values and apply offset
        mjd_filt = lc['MJD'][fmask]
        y_filt = lc['lum'][fmask] + offsets[filt]  # Apply offset
        y_filt_err = lc['dlum'][fmask]
        if pilcas:
            y_pilca = pilcas[i]
        # Select marker style based on index
        marker = markers[i % len(markers)]
        
        style = filt.plotstyle

        # Plot with error bars
        # ax.errorbar(mjd_filt, y_filt, yerr=y_filt_err,
        #  fmt=marker, label=f"{filt} (offset: {offsets[filt]:.1g})",
        #  capsize=3, **style)
        ax.plot(mjd_filt, np.log10(y_filt), 
         ls="-", label=f"{filt} (offset: {offsets[filt]:.1g})", **style)
        # print(filt)
        # aaa
        ax.plot(mjd_filt, torch.log10(y_pilca), c=style["mfc"], lw=2, ls="--")

    # Aesthetics
    ax.set_xlabel("MJD")
    ax.set_ylabel("Magnitude + Offset")
    # ax.invert_yaxis()  # Invert y-axis for magnitudes
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Light Curve Plot with Offsets")
    # ax.set_yscale("log")
    # Move the legend to the right of the plot
    ax.legend(title="Filter (Offset)", loc='center left', bbox_to_anchor=(1, 0.5))


def make_pilcas(outputs, lc,filter_for_mjd, MJD, filters_mask):
    model_inputs_torch = outputs.detach()#torch.tensor(model_inputs)#/units_array

    v_s = model_inputs_torch[0]       # Shock velocity
    M_env = model_inputs_torch[1]     # Envelope mass
    f_rho_M = model_inputs_torch[1]   # Density profile factor
    R = model_inputs_torch[2]         # Radius
    t_exp = model_inputs_torch[3]     # explosion time
    model = tp.ShockCooling4(z=lc.meta["redshift"])
    pilcas = []
    # fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    for i, f in enumerate(np.unique(filter_for_mjd)):
        mask = filter_for_mjd == f
        print(np.sum(mask), np.sum(lc["filter"]==f), np.sum(filters_mask.detach().numpy()[i]))
        y = model(MJD[mask], v_s=v_s, M_env=M_env, f_rho_M=f_rho_M, R=R, t_exp=t_exp, f=f)
        # print(len(y), np.sum(mask))
        # ax.plot(MJD[mask], y)
        pilcas.append(y)
    return pilcas


    # len(MJD), len(lc["MJD"]), np.sum(np.array(lc["filter"])==np.unique(filter_for_mjd)[0])

def plot_comparison(lc, outputs, filter_for_mjd, MJD, filters_mask):
    pilcas = make_pilcas(outputs, lc,filter_for_mjd, MJD, filters_mask)
    pilca_light_curve_plot(lc, offset = 1, pilcas=pilcas, ufilters=np.unique(filter_for_mjd))