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

torch.set_default_dtype(torch.float64)

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
    

class SC4Loss(nn.Module):
    def __init__(self, sc4model, ufilters):
        """
        Custom Weighted Mean Squared Error Loss
        :param weight: A tensor of weights for each sample (optional).
        """
        super(SC4Loss, self).__init__()
        self.sc4model = sc4model
        self.ufilters = ufilters
        # self.weight = weight

    def forward(self, outputs, targets, filters_mask):
        """
        Compute the weighted MSE loss.
        :param predictions: Model outputs (torch tensor).
        :param targets: Ground truth values (torch tensor).
        :return: Weighted MSE loss value.
        """
        # if (outputs < 0).any():
        #     return torch.tensor(float('inf'), device=outputs.device)
        # outputs = sigmoid(outputs, b=1e-3)
        # outputs = denormalize(outputs, low=0, high=10)
        # print(outputs)
        v_s = outputs[0]       # Shock velocity
        M_env = outputs[1]     # Envelope mass
        f_rho_M = outputs[1]   # Density profile factor
        R = outputs[2]         # Radius
        t_exp = outputs[3] 
        penalty = 1e-3*(torch.relu(t_exp - 3) ** 2) + 1e-4*(torch.relu(R - 6) ** 2) + 1e-4*(torch.relu(1/(M_env+0.1)))
        t_exp = torch.clip(t_exp, max=2.9999)
        # sigma = outputs[5] 
        loss = 0
        # print(targets[:,0][filters_mask[0]])
        for i, f in enumerate(self.ufilters):
            # print(f)
            lum = self.sc4model(targets[:,0][filters_mask[i]], v_s=v_s, M_env=M_env, f_rho_M=f_rho_M, R=R, t_exp=t_exp, f=f)

            y_fit = torch.log10(lum)
            y = targets[:,1][filters_mask[i]]
            # print(y[6], y_fit[6])
            # a
            # print((y-y_fit)**2/targets[:,2][filters_mask[i]]**2)
            # print(targets[:,2][filters_mask[i]])
            # print(y-y_fit)
            # aa
            loss += torch.sum((y-y_fit)**2)/len(y)#/targets[:,2][filters_mask[i]]**2))
            # print(loss)
            

        return loss+penalty#.mean( )+

class ModifiedSC4Loss(nn.Module):
    def __init__(self, sc4model, ufilters):
        """
        Custom Weighted Mean Squared Error Loss
        :param weight: A tensor of weights for each sample (optional).
        """
        super(ModifiedSC4Loss, self).__init__()
        self.sc4model = sc4model
        self.ufilters = ufilters
        # self.weight = weight

    def forward(self, outputs, targets, filters_mask):
        """
        Compute the weighted MSE loss.
        :param predictions: Model outputs (torch tensor).
        :param targets: Ground truth values (torch tensor).
        :return: Weighted MSE loss value.
        """
        # if (outputs < 0).any():
        #     return torch.tensor(float('inf'), device=outputs.device)
        # outputs = sigmoid(outputs, b=1e-3)
        # outputs = denormalize(outputs, low=0, high=10)
        # print(outputs)
        outputs = torch.clip(outputs,min=1e-3)
        v_s = outputs[0]       # Shock velocity
        M_env = outputs[1]     # Envelope mass
        f_rho_M = outputs[1]   # Density profile factor
        R = outputs[2]         # Radius
        t_exp = outputs[3] 
        # penalty = 1e-2*(torch.relu(t_exp - 3) ** 2) + 1e-4*(torch.relu(R - 6) ** 2) + 1e-4*(torch.relu(1/(M_env+0.1)))
        t_exp = torch.clip(t_exp, max=2.9999)
        t_exp = torch.clip(t_exp, min=1e-3) 
        
        # sigma = outputs[5] 
        loss = 0
        # print(targets[:,0][filters_mask[0]])
        y_fit_list = []
        for i, f in enumerate(self.ufilters):
            # print(f)
            lum = self.sc4model(targets[:,0][filters_mask[i]], v_s=v_s, M_env=M_env, f_rho_M=f_rho_M, R=R, t_exp=t_exp, f=f)

            y_fit_list.append(torch.log10(lum))

            # y = targets[:,1][filters_mask[i]]

            # loss += torch.sum((y-y_fit)**2)/len(y)
            
        y_fit = torch.concatenate(y_fit_list)
        return y_fit#loss+penalty#.mean( )+
    
    def get_y_true(self, targets, filters_mask):
        y = []
        for i, f in enumerate(self.ufilters):
            # print(f)
            # lum = self.sc4model(targets[:,0][filters_mask[i]], v_s=v_s, M_env=M_env, f_rho_M=f_rho_M, R=R, t_exp=t_exp, f=f)
            # print(targets[:,1][filters_mask[i]].shape)
            # aaa
            # y_fit.append(torch.log10(lum))
            y.append(targets[:,1][filters_mask[i]])
        y_true = torch.concatenate(y)
        return y_true

    # def get_penalty(self, outputs):
    #     v_s = outputs[0]       # Shock velocity
    #     M_env = outputs[1]     # Envelope mass
    #     f_rho_M = outputs[1]   # Density profile factor
    #     R = outputs[2]         # Radius
    #     t_exp = outputs[3] 
    #     penalty = 1e-5*(torch.relu(t_exp - 3) ** 2) #+ 1e-3*(torch.relu(R - 10) ** 2) + 1e-6*(torch.relu(1/(M_env+0.1)))
    #     return penalty
    def get_penalty(self, outputs):
        v_s, M_env, R, t_exp = outputs

        # Penalize out-of-bounds values
        penalty = 0.0 + 1e-2*(torch.relu(R - 6) ** 2)
        penalty += 1e-5 * (torch.relu(t_exp - 3) ** 2)      # upper bound for t_exp
        penalty += 1e-4 * (torch.relu(-v_s) ** 2)           # penalize v_s < 0
        penalty += 1e-4 * (torch.relu(-M_env) ** 2)         # penalize M_env < 0
        # penalty += 1e-4 * (torch.relu(-f_rho_M) ** 2)       # penalize f_rho_M < 0
        penalty += 1e-4 * (torch.relu(-R) ** 2)             # penalize R < 0
        penalty += 1e-4 * (torch.relu(-t_exp) ** 2)         # penalize t_exp < 0

        return penalty

def sigmoid(x, b):
    """Basic sigmoid function."""
    return 1 / (1 + torch.exp(-x*b))

def denormalize(x, low, high):
    """
    Desnormaliza valores de [0,1] para [low, high].

    Args:
        x (torch.Tensor): valores normalizados (no intervalo [0,1])
        low (float ou torch.Tensor): limite inferior do intervalo alvo
        high (float ou torch.Tensor): limite superior do intervalo alvo

    Returns:
        torch.Tensor: valores no intervalo [low, high]
    """
    return low + (high - low) * x

def get_physical(x, b, low, high):
    x = sigmoid(x, b)
    x = denormalize(x, low, high)
    return x

def print_outputs(outputs, loss, step):
    v_s, M_env,  R, t_exp = outputs  # unpack dos parâmetros
    f_rho_M = M_env
    # exemplo de cálculo da loss (substitua pelo seu)
    # loss = physics_loss(outputs, target)

    # imprime tudo de forma organizada
    print(f"--- Step {step} ---")
    print(f"Shock velocity (v_s):     {v_s.item():.4f}")
    print(f"Envelope mass (M_env):    {M_env.item():.4f}")
    print(f"Density factor (f_rho_M): {f_rho_M.item():.4f}")
    print(f"Radius (R):               {R.item():.4f}")
    print(f"Explosion time (t_exp):   {t_exp.item():.4f}")
    print(f"Loss:                     {loss.item():.6f}")
    print("-" * 40)


class MultiFilterMDN(nn.Module):
    def __init__(self, x_data, filters_mask, param_dim=5, num_components=3, hidden_dim=128):
        """
        MDN that handles variable input sizes per filter.
        Each filter gets its own branch sized according to how many samples it has.
        """
        super().__init__()
        self.param_dim = param_dim
        self.num_components = num_components
        self.num_filters = filters_mask.shape[0]
        self.filters_mask = filters_mask
        self.x_data = x_data

        # --- Create one branch per filter ---
        self.filter_nets = nn.ModuleList()
        for i_f in range(self.num_filters):
            n_points = int(torch.sum(self.filters_mask[i_f]).item())
            # Each point has 3 features: [MJD, LUM, DLUM]
            input_dim = n_points * self.x_data.shape[1]
            self.filter_nets.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
            )

        # --- Combine all filters ---
        combined_dim = hidden_dim * self.num_filters
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU()
        )

        # --- MDN heads ---
        self.pi_head = nn.Linear(hidden_dim, num_components)
        self.mu_head = nn.Linear(hidden_dim, num_components * param_dim)
        self.sigma_head = nn.Linear(hidden_dim, num_components * param_dim)

    def forward(self, ):
        """
        x: [N_points, N_features]  (the full light curve data)
        filters_mask: [N_filters, N_points] boolean mask
        """
        filter_latents = []
        x = self.x_data
        for i_f, net in enumerate(self.filter_nets):
            # select data points belonging to this filter
            x_f = x[self.filters_mask[i_f]]  # [n_points_i, n_features]
            # flatten: concatenate all time samples for this filter
            x_f_flat = x_f.flatten().unsqueeze(0)  # shape [1, n_points_i * n_features]
            h_f = net(x_f_flat)
            filter_latents.append(h_f)

        # concatenate all filters' latent features
        h_all = torch.cat(filter_latents, dim=-1)

        # combine
        h = self.combined_net(h_all)
        pi = F.softmax(self.pi_head(h), dim=-1)
        mu = F.relu(self.mu_head(h)).view(-1, self.num_components, self.param_dim)
        sigma = F.softplus(self.sigma_head(h)) + 1e-5
        sigma = sigma.view(-1, self.num_components, self.param_dim)

        return pi, mu, sigma

    def log_prob(self, y_true):
        """
        Compute mean log-likelihood of y_true under the MDN distribution.
        """
        x = self.x_data
        filters_mask = self.filters_mask

        pi, mu, sigma = self.forward(x, filters_mask)
        y = y_true.unsqueeze(1).expand_as(mu)
        gauss = -0.5 * (((y - mu) / sigma) ** 2 + 2 * torch.log(sigma) + torch.log(torch.tensor(2 * torch.pi)))
        log_prob = torch.logsumexp(torch.log(pi) + gauss.sum(dim=-1), dim=-1)
        return log_prob.mean()

    def get_params(self,):
        pi, mu, sigmas = self()

        mean_params = torch.zeros(5)
        for i in range(3):
            mean_params += pi[0,i]*mu[0,i] 
        # for i in len(self.num_components):
        #     mean_params = pi[0,i]*mu[0,i] 


        return mean_params


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_var=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var

        # variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).fill_(-3))

        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        # reparameterization trick
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        eps_w = torch.randn_like(weight_sigma)
        eps_b = torch.randn_like(bias_sigma)

        weight = self.weight_mu + weight_sigma * eps_w
        bias = self.bias_mu + bias_sigma * eps_b

        # log probs for ELBO
        prior = torch.distributions.Normal(0, self.prior_var**0.5)
        var_post_w = torch.distributions.Normal(self.weight_mu, weight_sigma)
        var_post_b = torch.distributions.Normal(self.bias_mu, bias_sigma)

        self.log_prior = prior.log_prob(weight).sum() + prior.log_prob(bias).sum()
        self.log_variational_posterior = var_post_w.log_prob(weight).sum() + var_post_b.log_prob(bias).sum()

        return F.linear(x, weight, bias)


class MultiFilterBNN(nn.Module):
    def __init__(self, x_data, filters_mask, param_dim=5, hidden_dim=64):
        super().__init__()
        self.param_dim = param_dim
        self.num_filters = filters_mask.shape[0]
        self.filters_mask = filters_mask
        self.x_data = x_data

        self.filter_nets = nn.ModuleList()
        for i_f in range(self.num_filters):
            n_points = int(torch.sum(self.filters_mask[i_f]).item())
            input_dim = n_points * self.x_data.shape[1]
            self.filter_nets.append(
                nn.Sequential(
                    BayesianLinear(input_dim, hidden_dim),
                    nn.ReLU(),
                    BayesianLinear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
            )

        combined_dim = hidden_dim * self.num_filters
        self.output_net = nn.Sequential(
            BayesianLinear(combined_dim, hidden_dim),
            nn.ReLU(),
            BayesianLinear(hidden_dim, param_dim),
            PositiveLeakyReLU(alpha=1e-1, epsilon=2),
            nn.ReLU()
            
            # nn.Linear(param_dim, param_dim),
            # nn.ReLU()
        )

    def forward(self):
        x = self.x_data
        latents = []
        for i_f, net in enumerate(self.filter_nets):
            x_f = x[self.filters_mask[i_f]]
            x_f_flat = x_f.flatten().unsqueeze(0)
            h_f = net(x_f_flat)
            latents.append(h_f)

        h_all = torch.cat(latents, dim=-1)
        out = self.output_net(h_all)
        return out.squeeze(0)


class PositiveLeakyReLU(nn.Module):
    """
    Activation similar to LeakyReLU but ensures strictly positive outputs.
    f(x) = max(αx, x) + ε
    """
    def __init__(self, alpha=0.01, epsilon=1e-3):
        super().__init__()
        self.negative_slope = alpha
        self.epsilon = epsilon

    def forward(self, x):
        # LeakyReLU behavior + epsilon shift to ensure positivity
        return F.leaky_relu(x, negative_slope=self.negative_slope) + self.epsilon