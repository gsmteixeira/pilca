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


k_B = const.k_B.to("eV / kK").value
c1 = (const.h / const.k_B).to(u.kK / u.THz).value
c2 = 8 * np.pi ** 2 * (const.h / const.c ** 2).to(u.W / u.Hz / (1000 * u.Rsun) ** 2 / u.THz ** 3).value
c = const.c.to(u.angstrom * u.THz).value

c3 = (4. * np.pi * const.sigma_sb.to("erg s-1 Rsun-2 kK-4").value) ** -0.5 / 1000.  # Rsun --> kiloRsun
c4 = 1. / (4. * np.pi * u.Mpc.to(u.m) ** 2.)

def power(base, exp):
    """Power function that returns zero for any nonpositive base"""
    base, exp = torch.as_tensor(base), torch.as_tensor(exp)
    broadcast_shape = torch.broadcast_shapes(base.shape, exp.shape)
    
    zeros = torch.zeros(broadcast_shape, dtype=torch.float32)
    positive = base > 0
    power = torch.pow(base, exp)
    
    return torch.where(positive, power, zeros)

def blackbody_to_filters(f, T, R, z=0., cutoff_freq=torch.inf, ebv=0.):

    # T = torch.as_tensor(T)  
    # R = torch.as_tensor(R)
    
    if T.shape != R.shape:
        raise Exception('T & R must have the same shape')

    # Check if T and ebv are broadcastable
    # torch.broadcast_shapes(T.shape, torch.as_tensor(ebv).shape)

    # if T.ndim == 1 and len(T) == len(filters):  # pointwise case
    #     y_fit = torch.stack([f.synthesize(planck_fast, t, r, cutoff_freq, z=z, ebv=ebv) 
    #                          for f, t, r in zip(filters, T, R)])
    # else:
    y_fit = synthesize(f, planck_fast, T, R, cutoff_freq, z=z, ebv=ebv) 

    return y_fit

def extinction_law(freq, ebv, rv=3.1):
    
    freq = torch.as_tensor(freq)
    ebv = torch.atleast_1d(torch.as_tensor(ebv))
    
    A = torch.stack([torch.tensor(fitzpatrick99(c / freq.detach().numpy(), rv * e, rv)) for e in ebv]).requires_grad_(False) 
    # A = np.squeeze([fitzpatrick99(c / freq, rv * e, rv) for e in np.atleast_1d(ebv)])
    return 10. ** (A / -2.5)

def planck_fast(nu, T, R, cutoff_freq=torch.inf):
    
    R_squared_outer = torch.outer(R**2, torch.ones_like(nu))
    nu_terms = torch.outer(torch.ones_like(R), nu**3 * torch.minimum(torch.tensor(1.0), cutoff_freq / nu))
    
    # Compute power and exponential terms
    exp_term = torch.exp(c1 * torch.outer(T.reciprocal(), nu)) - 1.
    power_term = power(exp_term, -1.)
    
    # Final computation
    result = c2 * torch.squeeze(R_squared_outer * nu_terms * power_term)

    return result


def synthesize(f, spectrum, *args, z=0., ebv=0., **kwargs):
        
        freq = torch.tensor(f.trans['freq'].value * (1. + z)).requires_grad_(False) 
        # print(spectrum(freq, *args, **kwargs) )
        # aaa
        return torch.trapz(spectrum(freq, *args, **kwargs) * extinction_law(freq, ebv)
                        * torch.tensor(f.trans['T_norm_per_freq'].data), torch.tensor(f.trans['freq'].data, requires_grad=False))

class ShockCooling4():
    """
    The shock cooling model of Morag, Sapir, & Waxman (https://doi.org/10.1093/mnras/stad899).

    :math:`L(\\tilde{t}) = L_\\mathrm{br}\\left\{\\tilde{t}^{-4/3} + 0.9\\exp\\left[-\\left(\\frac{2.0t}{t_\\mathrm{tr}}\\right)^{0.5}\\right] \\tilde{t}^{-0.17}\\right\}` (Eq. A1)

    :math:`T_\\mathrm{col}(\\tilde{t}) = T_\\mathrm{col,br} \\min(0.97\\tilde{t}^{-1/3}, \\tilde{t}^{-0.45})` (Eq. A2)

    :math:`\\tilde{t} \\equiv \\frac{t}{t_\\mathrm{br}}`, where :math:`t_\\mathrm{br} = (0.86\\,\\mathrm{h}) R^{1.26} v_\\mathrm{s*}^{-1.13} (f_\\rho M \\kappa)^{-0.13}` (Eq. A5)

    :math:`L_\\mathrm{br} = (3.69 \\times 10^{42}\\,\\mathrm{erg}\\,\\mathrm{s}^{-1}) R^{0.78} v_\\mathrm{s*}^{2.11} (f_\\rho M)^{0.11} \\kappa^{-0.89}` (Eq. A6)

    :math:`T_\\mathrm{col,br} = (8.19\\,\\mathrm{eV}) R^{-0.32} v_\\mathrm{s*}^{0.58} (f_\\rho M)^{0.03} \\kappa^{-0.22}` (Eq. A7)

    :math:`t_\\mathrm{tr} = (19.5\\,\\mathrm{d}) \\sqrt{\\frac{\\kappa M}{v_\\mathrm{s*}}}` (Eq. A9)

    Parameters
    ----------
    lc : lightcurve_fitting.lightcurve.LC, optional
        The light curve to which the model will be fit. Only used to get the redshift if `redshift` is not given.
    redshift : float, optional
        The redshift between blackbody source and the observed filters. Default: 0.

    Attributes
    ----------
    z : float
        The redshift between blackbody source and the observed filters
    n : float
        The polytropic index of the progenitor
    A : float
        Coefficient on the luminosity suppression factor (Eq. A1)
    a : float
        Coefficient on the transparency timescale (Eq. A1)
    alpha : float
        Exponent on the transparency timescale (Eq. A1)
    L_br_0 : float
        Coefficient on the luminosity expression in erg/s (Eq. A6)
    T_col_br_0 : float
        Coefficient on the temperature expression in eV (Eq. A7)
    t_min_0 : float
        Coefficient on the minimum validity time in days (Eq. A3)
    t_br_0 : float
        Coefficient on the :math:`\\tilde{t}` timescale in days (Eq. A5)
    t_07eV_0 : float
        Coefficient on the time at which the ejecta reach 0.7 eV in days (Eq. A8)
    t_tr_0 : float
        Coefficient on the transparency timescale in days (Eq. A9)
    """
    input_names = [
        'v_\\mathrm{s*}',
        'M_\\mathrm{env}',
        'f_\\rho M',
        'R',
        't_0',
    ]
    units = [
        10. ** 8.5 * u.cm / u.s,
        u.Msun,
        u.Msun,
        1e13 * u.cm,
        u.d,
    ]

    def __init__(self, z):
        # super().__init__(lc, redshift=redshift)
        self.z = z
        self.A = 0.9
        self.a = 2.
        self.alpha = 0.5
        self.L_br_0 = 3.69e42  # erg / s
        self.T_col_br_0 = 8.19  # eV
        self.t_min_0 = 0.012  # d (17 min)
        self.t_br_0 = 0.036  # d (0.86 h)
        self.t_07eV_0 = 6.86  # d
        self.t_tr_0 = 19.5  # d

    def temperature_radius(self, t_in, v_s, M_env, f_rho_M, R, t_exp=0., kappa=1.):
        t_br = self.t_br_0 * R ** 1.26 * v_s ** -1.13 * f_rho_M ** -0.13  # Eq. A5
        # t_br = self.t_br_0 * R ** 1.26 * v_s ** -1.13 * f_rho_M ** -0.13  # Eq. A5

        L_br = self.L_br_0 * R ** 0.78 * v_s ** 2.11 * f_rho_M ** 0.11 * kappa ** -0.89  # Eq. A6
        # L_br = self.L_br_0 * R ** 0.78 * v_s ** 2.11 * f_rho_M ** 0.11 * kappa ** -0.89  # Eq. A6
        # L_br = self.L_br_0 * R ** 0.78 * v_s ** 2.11 * f_rho_M ** 0.11 * kappa ** -0.89  # Eq. A6

        T_col_br = self.T_col_br_0 * R ** -0.32 * v_s ** 0.58 ** f_rho_M ** 0.03 * kappa ** -0.22  # Eq. A7
        # T_col_br = self.T_col_br_0 * R ** -0.32 * v_s ** 0.58 ** f_rho_M ** 0.03 * kappa ** -0.22  # Eq. A7

        t_tr = self.t_tr_0 * torch.sqrt(kappa * M_env / v_s)  # Eq. A9
        # t_tr = self.t_tr_0 * np.sqrt(kappa * M_env / v_s)  # Eq. A9

        t = torch.reshape(torch.as_tensor(t_in), (-1, 1)) - t_exp

        ttilde = t / t_br

        L = L_br * (power(ttilde, -4. / 3.) +
                    self.A * torch.exp(-power(self.a * t / t_tr, self.alpha)) * power(ttilde, -0.17))  # Eq. A1
        # L = L_br * (power(ttilde, -4. / 3.) +
        #             self.A * np.exp(-power(self.a * t / t_tr, self.alpha)) * power(ttilde, -0.17))
        # L = L_br * (power(ttilde, -4. / 3.) +
        #             self.A * np.exp(-power(self.a * t / t_tr, self.alpha)) * power(ttilde, -0.17))
        
        T_col = T_col_br * torch.minimum(0.97 * power(ttilde, -1. / 3.), power(ttilde, -0.45))  # Eq. A2
        # T_col = T_col_br * np.minimum(0.97 * power(ttilde, -1. / 3.), power(ttilde, -0.45))  # Eq. /A2

        T_K = torch.squeeze(T_col) / k_B
        R_bb = c3 * torch.squeeze(L) ** 0.5 * power(T_K, -2.)
        # R_bb = c3 * np.squeeze(L) ** 0.5 * power(T_K, -2.)

        # T_K = np.squeeze(T_col) / k_B
        # R_bb = c3 * np.squeeze(L) ** 0.5 * power(T_K, -2.)

        return T_K, R_bb#self.L_br_0 * R #torch.squeeze(L)#R_bb

    def __call__(self, t_in, f, v_s, M_env, f_rho_M, R, t_exp=0., kappa=1.):
        """
        Evaluate this model at a range of times and filters

        Parameters
        ----------
        t_in : float, array-like
            Time in days
        f : lightcurve_fitting.filter.Filter, array-like
            Filters for which to calculate the model
        v_s : float, array-like
            The shock speed in :math:`10^{8.5}` cm/s
        M_env : float, array-like
            The envelope mass in solar masses
        f_rho_M : float, array-like
            The product :math:`f_ρ M`, where ":math:`f_ρ` is a numerical factor of order unity that depends on the inner
            envelope structure" and :math:`M` is the ejecta mass in solar masses
        R : float, array-like
            The progenitor radius in :math:`10^{13}` cm
        t_exp : float, array-like, optional
            The explosion epoch. Default: 0.
        kappa : float, array-like, optional
            The ejecta opacity in units of the electron scattering opacity (0.34 cm^2/g). Default: 1.

        Returns
        -------
        y_fit : array-like
            The filtered model light curves
        """
        T_K, R_bb = self.temperature_radius(t_in, v_s, M_env, f_rho_M, R, t_exp, kappa)

        # return R_bb

        lum_blackbody = blackbody_to_filters(f, T_K, R_bb, self.z)
        lum_suppressed = blackbody_to_filters(f, 0.74 * T_K, 0.74 ** -2. * R_bb, self.z)
        lum = torch.minimum(lum_blackbody, lum_suppressed)  # Eq. A4
        return lum

    def t_min(self, p, kappa=1.):
        """
        The minimum time at which the model is valid

        :math:`t_\\mathrm{min} = (17\\,\\mathrm{min}) R + t_\\mathrm{exp}` (Eq. A3)
        """
        R = p[3]
        t_exp = p[4] if len(p) > 4 else 0.
        return self.t_min_0 * R + t_exp  # Eq. A3

    def t_max(self, p, kappa=1.):
        """
        The maximum time at which the model is valid

        :math:`t_\\mathrm{max} = \\min(t_\\mathrm{0.7\\,eV}, 0.5 t_\\mathrm{tr})` (Eq. A3)

        :math:`t_\\mathrm{0.7\\,eV} = (6.86\\,\\mathrm{d}) R^{0.56} v_\\mathrm{s*}^{0.16} \\kappa^{-0.61} (f_\\rho M)^{-0.06}` (Eq. A8)

        :math:`t_\\mathrm{tr} = (19.5\\,\\mathrm{d}) \\sqrt{\\frac{\\kappa M}{v_\\mathrm{s*}}}` (Eq. A9)
        """
        v_s, M_env, f_rho_M, R, t_exp, *_ = p
        t_07eV = self.t_07eV_0 * R ** 0.56 * v_s ** 0.16 * kappa ** -0.61 * f_rho_M ** -0.06  # Eq. A8
        t_tr = self.t_tr_0 ** np.sqrt(kappa * M_env / v_s)  # Eq. A9
        return np.minimum(t_07eV, t_tr / self.a) + t_exp  # Eq. A3
    

