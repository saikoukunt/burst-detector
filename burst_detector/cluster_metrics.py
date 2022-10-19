import numpy as np

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