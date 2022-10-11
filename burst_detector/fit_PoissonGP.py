import numpy as np, scipy as sp
from mk_GP_mat import *

def xFit = fit_PoissonGP(t_spikes, x_params, tLims, dt)
    
    t_toFit = np.linspace(min(tLims), max(tLims), math.ceil((max(tLims)-min(tLims))/dt))
    llFunc = lambda x: mk_AJCP_ll(x, t_spikes, t_toFit, dt, x_params)
    
    # run GP fit
    
    # ???

def mk_AJCP_ll(x, t_spikes, t_toFit, dt, x_params, nargout=1):
    # Input parsing
    # ensure that x is a vector
    tol = .01
    avWt = 2*np.ones_like(x)
    avWt[[0,-1]] = 1
    avWt = avWt*dt/2
    
    # Add in PSTH-dependent portions
    Kxxb = mk_GP_mat(t_spikes, t_toFit, x_params) # GP cross-cov matrix between spike times and summary points
    Kxbxb = mk_GP_mat(t_toFit, t_toFit, x_params) # GP auto-cov matrix for summary points
    
    Kxxb_sum = Kxxb.sum(0) # Pre-calculate cross-cov sum
    KIxbxb = np.linalg.pinv(Kxbxb, tol) # Pre-calculate pseudoinverse
    
    LL = 0.5*np.sum(avWt*np.exp(x))
    LL = LL - Kxxb_sum*(KIxbxb*x) + 0.5*x.T*(KIxbxb*x)
    
    if nargout > 1:
        gradLL = -KIxbxb.T*Kxxb_sum.T + KIxbxb*x + 0.5*avWt*np.exp(x)
        return LL, gradLL
        
    if nargout > 2: 
        hessLL = KIxbxb + np.diag(0.5*avWt*np.exp(x))
        return LL, gradLL, hessLL
    
    return LL