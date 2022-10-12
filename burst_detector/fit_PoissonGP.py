import numpy as np, scipy as sp
from burst_detector import *
import math
from scipy.optimize import fmin_ncg

def fit_PoissonGP(t_spikes, x_params, tLims, dt):
    
    t_toFit = np.linspace(min(tLims), max(tLims), math.ceil((max(tLims)-min(tLims))/dt))
    avWt = np.zeros_like(t_toFit)
    
    llFunc = lambda x: mk_AJCP_ll(x, t_spikes, t_toFit, dt, x_params)
    gradFunc = lambda x: get_gradLL(x, t_spikes, t_toFit, dt, x_params)
    hessFunc = lambda x: get_hessLL(x, t_spikes, t_toFit, dt, x_params)
    
    # run GP fit
    xopt = fmin_ncg(llFunc, np.zeros_like(t_toFit), gradFunc, fhess= hessFunc, maxiter=100)
    
    return xopt

def mk_AJCP_ll(x, t_spikes, t_toFit, dt, x_params, nargout=1):
    # Input parsing
    # ensure that x is a vector
    tol = .01
    
    global avWt
    avWt = 2*np.ones_like(x)
    avWt[[0,-1]] = 1
    avWt = avWt*dt/2
    
    # Add in PSTH-dependent portions
    Kxxb = mk_GP_mat(t_spikes, t_toFit, x_params) # GP cross-cov matrix between spike times and summary points
    global Kxbxb 
    Kxbxb = mk_GP_mat(t_toFit, t_toFit, x_params) # GP auto-cov matrix for summary points
    
    global Kxxb_sum 
    Kxxb_sum = Kxxb.sum(0) # Pre-calculate cross-cov sum
    global KIxbxb 
    KIxbxb = np.linalg.pinv(Kxbxb, tol) # Pre-calculate pseudoinverse
    
    LL = 0.5*np.sum(avWt*np.exp(x))
    LL = LL - Kxxb_sum@(KIxbxb@x) + 0.5*x.T@(KIxbxb@x)
    
    return LL

def get_gradLL(x, t_spikes, t_toFit, dt, x_params):
    
    gradLL = -KIxbxb.T@Kxxb_sum.T + KIxbxb@x + 0.5*avWt*np.exp(x)
    
    return gradLL

def get_hessLL(x, t_spikes, t_toFit, dt, x_params):
    
    hessLL = KIxbxb + np.diag(0.5*avWt*np.exp(x))
    return hessLL