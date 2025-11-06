from lightcurve_fitting_loglike_fpm import lightcurve, models, fitting, bolometric
from pkg_resources import resource_filename
from IPython.display import display, Math
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter 
from scipy import interpolate as interp
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize, curve_fit


def main():

    N_STEPS_BURNIN = 1
    N_STEPS = 10000
    N_WALKERS = 40

    lc = prepare_lc()
    # print(lc)
    min_mjd = np.min(lc['MJD'])
    T_max = 59944.44#59931.54#59935.27#59931.92#59928.92
    
    lc_early = lc.where(MJD_min=min_mjd-1, MJD_max=T_max) 

    model = models.ShockCooling4(lc_early)


    ten13cmtoRsol = 1e13*1.4374e-11 
    ten8p5cmstoten3kms = 10**.5
    # units_array = np.array([ten8p5cmstoten3kms, 1, 1, ten13cmtoRsol, 1, 1]).reshape(1,-1)

    priors = [
    models.UniformPrior(0/ten8p5cmstoten3kms, 10/ten8p5cmstoten3kms),
    models.UniformPrior(0, 10),
    models.UniformPrior(1e-3, 100.),
    models.UniformPrior(0, 14374),
    models.UniformPrior(59910, 59920.4333),
    models.LogUniformPrior(0,1e2),
    ]

    p_lo = [0.8/ten8p5cmstoten3kms, 1.5, 0.4*2.8, 600/ten13cmtoRsol, 59913, 6]
    p_up = [1.1/ten8p5cmstoten3kms, 2.8, 0.4*2.8+.2, 680/ten13cmtoRsol, 59920, 8]

    sampler = fitting.lightcurve_mcmc(lc_early, model, 
                                    priors=priors, p_lo=p_lo, p_up=p_up,
                                    nwalkers=N_WALKERS, nsteps=N_STEPS, nsteps_burnin=N_STEPS_BURNIN, use_sigma=True)

    np.save('outputs/sc4_flattchain_DYNAMICAL_fpm_newtmax.npy', np.array(sampler.flatchain))

def prepare_lc():
    data_dir = os.path.join('/tf/ProjectGabriel/SUPERNOVAE/data/','2022acko')

    filename = os.path.join(data_dir,'2022acko_total.dat')
    lc = lightcurve.LC.read(filename)

    lc.meta['dm'] = 30.81
    lc.meta['extinction'] = {
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
    'w': 0.071,
    'y': 0.033,
    'UVW2': 0.234,
    'UVM2': 0.211,
    'UVW1': 0.166,
    'DLT40': 0.083,
    }
    # Based on no Na I D detection in spectrum
    lc.meta['hostext'] = {
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
    'w': 0.,
    'y': 0.,
    }
    z = 0.00526
    lc.meta['redshift'] = z  # redshift
    lc.calcAbsMag()
    lc.calcFlux()
    return lc


if __name__=='__main__':
    main()

