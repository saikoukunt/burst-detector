import numpy as np
import burst_detector as bd
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_wfs(wfs):
    """
    Normalizes a each waveform in an array of waveforms with respect to the Frobenius norm.
    
    Parameters
    ----------
    wfs: array-like
        Array containing waveforms. The first dimension of the array must index the 
        waveforms [i.e. shape = (# of waveforms, # channels, # timepoints) OR 
        (# of waveforms, # timepoints, # channels)]
        
    Returns
    -------
    wfs_norm: array-like
        Array of waveforms where each waveform has been normalized to have a Frobenius norm
        of 1. Has same shape as input
    """
    n_wf = wfs.shape[0]
    
    wfs_norm = wfs.copy()
    
    for i in range(n_wf):
        norm = np.linalg.norm(wfs[i])
        if norm > 0:
            wfs_norm[i] /= norm
            
    return wfs_norm


def wf_means_similarity(mean_wf):
    """
    Calculates the normalized pairwise similarity (inner product) between every pair of 
    waveforms in an array of waveforms.
    
    Parameters
    ----------
    mean_wf: array-like
        Array containing waveforms. The first dimension of the array must index the 
        waveforms [i.e. shape = (# of waveforms, # channels, # timepoints) OR 
        (# of waveforms, # timepoints, # channels)]
    
    Returns
    -------
    mean_sim: array-like
        A symmetric matrix containing the pairwise similarity for each pair of (normalized) 
        waveforms.
    mean_wf_norm: array-like
        Array containing normalized input waveforms, where each waveform is normalized with
        respect to its Frobenius norm.
    """
    n_clust = mean_wf.shape[0]
    mean_sim = np.zeros((n_clust, n_clust))
    
    mean_wf_norm = normalize_wfs(mean_wf)
        
    for i in range(n_clust):
        for j in range(n_clust):
            if i != j:
                mean_sim[i, j] = np.dot(mean_wf_norm[i].flatten(), mean_wf_norm[j].flatten())
        
    return mean_sim, mean_wf_norm


def cross_proj(c1_spikes, c2_spikes, c1_mean_norm, c2_mean_norm, norm=True):
    """
    Returns the cross-projections of spikes onto normalized mean waveforms for a 
    pair of clusters.
    
    Parameters
    ----------
    c1_spikes, c2_spikes: int or array-like
        An array of spikes in the specified cluster with shape (# of spikes, # of channels, 
        # of timepoints).
    c1_mean_norm, c2_mean_norm: array_like
        The normalized mean waveform for the respective cluster. Should have shape (# of channels, # of 
        timepoints)
    norm: boolean, optional
        If True, spike waveforms are normalized with respect to their Frobenius norms
        before projection. True by default.
        
    Returns
    -------
    proj_1on1, proj_2on1, proj_1on2, proj_2on2: array-like
        1-D arrays containing the projections of spikes onto mean waveforms. 
    """    
    
    if norm:
        c1_spikes = normalize_wfs(c1_spikes)
        c2_spikes = normalize_wfs(c2_spikes)
        
    proj_1on1 = np.zeros((c1_spikes.shape[0]))
    proj_2on1 = np.zeros((c2_spikes.shape[0]))
    
    proj_1on2 = np.zeros((c1_spikes.shape[0]))
    proj_2on2 = np.zeros((c2_spikes.shape[0]))
    
    for i in range(c1_spikes.shape[0]):
        proj_1on1[i] = np.dot(c1_spikes[i].flatten(), c1_mean_norm.flatten())
        proj_1on2[i] = np.dot(c1_spikes[i].flatten(), c2_mean_norm.flatten())
        
    for i in range(c2_spikes.shape[0]):
        proj_2on1[i] = np.dot(c2_spikes[i].flatten(), c1_mean_norm.flatten())
        proj_2on2[i] = np.dot(c2_spikes[i].flatten(), c2_mean_norm.flatten())
        
    return proj_1on1, proj_2on1, proj_1on2, proj_2on2


def plot_cross_proj(proj_1on1, proj_2on1, proj_1on2, proj_2on2, id_1, id_2, bin_width=.01):
    """
    Plots histograms of cross-projection values.
    
    Parameters
    ----------
    proj_1on1, proj_2on1, proj_1on2, proj_2on2: array-like
        1-D arrays containing the cross-projections (i.e. output of cross_proj)
    bin_width: float
        Width of histogram bins in seconds
    
    """
    
    bins = np.arange(0,.5,bin_width)
    
    plt.figure(figsize=(6,2))
    
    plt.subplot(1,2,1)
    plt.title("Hist of projections onto cluster %d avg"%id_1, fontdict = {'fontsize' : 8})
    sns.histplot(proj_1on1, bins=bins)
    sns.histplot(proj_2on1, bins=bins)
    
    plt.subplot(1,2,2)
    plt.title("Hist of projections onto cluster %d avg"%id_2, fontdict = {'fontsize' : 8})
    sns.histplot(proj_1on2, bins=bins)
    sns.histplot(proj_2on2, bins=bins)
    
    plt.tight_layout()
    

def hist_cross_proj(proj_1on1, proj_2on1, proj_1on2, proj_2on2, bin_width=.01):
    """
    Calculates a histogram of input cross-projection values.
    
    Parameters
    ----------
    proj_1on1, proj_2on1, proj_1on2, proj_2on2: array-like
        1-D arrays containing the cross-projections (i.e. output of cross_proj)
    bin_width: float
        Width of histogram bins in seconds
        
    Returns
    -------
    hist_1on1, hist_2on1, hist_1on2, hist_2on2: array-like
        1-D arrays containing the histograms
    
    """
    
    bins = np.arange(0,1,bin_width)
    
    hist_1on1 = np.histogram(proj_1on1, bins=bins)[0]
    hist_2on1 = np.histogram(proj_2on1, bins=bins)[0]
    hist_1on2 = np.histogram(proj_1on2, bins=bins)[0]
    hist_2on2 = np.histogram(proj_2on2, bins=bins)[0]
    
    return hist_1on1, hist_2on1, hist_1on2, hist_2on2

    
def prob_cross_proj(proj_1on1, proj_2on1, proj_1on2, proj_2on2, bin_width=.01):
    """
    Calculates a histogram-based probability distribution on input cross-projection values.
    
    Parameters
    ----------
    proj_1on1, proj_2on1, proj_1on2, proj_2on2: array-like
        1-D arrays containing the cross-projections (i.e. output of cross_proj)
    bin_width: float
        Width of histogram bins in seconds
        
    Returns
    -------
    prob_1on1, prob_2on1, prob_1on2, prob_2on2: array-like
        1-D arrays containing the histograms
    
    """
    
    hist_1on1, hist_2on1, hist_1on2, hist_2on2 = hist_cross_proj(proj_1on1, proj_2on1, proj_1on2, proj_2on2, bin_width)
    
    prob_1on1 = hist_1on1/np.sum(hist_1on1)
    prob_2on1 = hist_2on1/np.sum(hist_1on1)
    prob_1on2 = hist_1on2/np.sum(hist_1on1)
    prob_2on2 = hist_2on2/np.sum(hist_1on1)
    
    return prob_1on1, prob_2on1, prob_1on2, prob_2on2
    
    
    
        
    