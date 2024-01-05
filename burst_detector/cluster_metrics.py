import numpy as np
from numpy.typing import NDArray
import burst_detector as bd
import matplotlib.pyplot as plt
import seaborn as sns

def calc_wf_norms(wfs: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Calculates the Frobenius norm of each waveform in an array of waveforms.
    
    Parameters
    ----------
    wfs: array-like
        Array containing waveforms. The first dimension of the array must index the 
        waveforms [i.e. shape = (# of waveforms, # channels, # timepoints) OR 
        (# of waveforms, # timepoints, # channels)]
        
    Returns
    -------
    wf_norm: array-like
        Array of waveforms where each waveform has been normalized to have a Frobenius norm
        of 1. Has same shape as input
    """
    wf_norms:  NDArray[np.float_] = np.zeros(wfs.shape[0])
    
    for i in range(wfs.shape[0]):
        wf_norms[i] = np.linalg.norm(wfs[i])
            
    return wf_norms


def wf_means_similarity(
        mean_wf: NDArray[np.float_],
        cl_good: NDArray[np.bool_], 
        jitter: bool = False, 
        jitter_amt: int = 4
    ) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.int_]]:
    """
    Calculates the normalized pairwise similarity (inner product) between pairs of 
    waveforms in an array of waveforms.
    
    Parameters
    ----------
    mean_wf: array-like
        Array containing waveforms. The first dimension of the array must index the 
        waveforms [i.e. shape = (# of waveforms, # channels, # timepoints) OR 
        (# of waveforms, # timepoints, # channels)]
    cl_good: array-like
        Array containing cluster quality status. Only good clusters will be considered
        in similarity calculations.
    jitter: boolean, optional
        True if similarity calculations should check for time shifts between waveforms.
        Default is False.
    jitter_amt: int, optional
        For time shift checking, number of samples to search in each direction. Default
        is 4 samples.
        
    Returns
    -------
    mean_sim: array-like
        A symmetric matrix containing the (maximum) pairwise similarity for each pair of (normalized) 
        waveforms.
    wf_norms: array-like
        Array containing normalized input waveforms, where each waveform is normalized with
        respect to its Frobenius norm.
    offsets: array-like
        Offset that produces that maximum inner product between waveforms
    """
    n_clust: int = mean_wf.shape[0]
    mean_sim: NDArray[np.float_] = np.zeros((n_clust, n_clust))
    offsets: NDArray[np.int_] = np.zeros((n_clust, n_clust), dtype='int16')
    wf_norms: NDArray[np.float_] = calc_wf_norms(mean_wf)
    
    for i in range(n_clust):
        for j in range(n_clust):
            if cl_good[i] and cl_good[j]:
                if i != j:
                    norm: NDArray[np.float_] = max(wf_norms[i], wf_norms[j])
                    if norm == 0:
                        continue

                    if jitter:
                        sim: float; off: int
                        sim, off = sim_jitter(mean_wf[i], mean_wf[j], jitter_amt)
                        mean_sim[i, j] = sim/(norm**2)
                        offsets[i,j] = off
                    else:
                        mean_sim[i, j] = np.dot(mean_wf[i].flatten(), mean_wf[j].flatten())/(norm**2)
                    
    return mean_sim, wf_norms, offsets


def sim_jitter(m1: NDArray[np.float_], m2: NDArray[np.float_], jitter_amt: int) -> tuple[float, int]:
    """
    Calculates the maximum inner product between two waveforms with respect to a time shift.
    
    Parameters
    ----------
    m1, m2: array-like
        2-D input waveforms of shape (# of channels, # of timepoints)
    jitter_amt: int
        Number of samples to time-shift in each direction.
        
    Returns
    -------
    mean_sim: float
        The computed maximum inner product.
    offset: int
        The time-shift offset that produced the maximum inner product. Specifies the amount that
        m2 shifted (e.g. offset=-1 means m2 was shifted back 1 sample)
    
    """
    offset = 0
    mean_sim: float = np.dot(m1.flatten(),m2.flatten())
    t_length: int = m1.shape[1]

    for i in range(-1*jitter_amt, jitter_amt+1):
        if i < 0:
            c1_shift: NDArray[np.float_] = m1[:,:t_length+i]
            c2_shift: NDArray[np.float_] = m2[:,i*-1:]
        else:
            c1_shift = m1[:,i:]
            c2_shift: NDArray[np.float_] = m2[:,:t_length-i]

        off_sim: float = np.dot(c1_shift.flatten(), c2_shift.flatten())
        if off_sim > mean_sim:
            mean_sim = off_sim
            offset: int = i

        if (mean_sim - np.dot(m1.flatten(),m2.flatten())) < .1:
            mean_sim = np.dot(m1.flatten(),m2.flatten())
            offset = 0
            
    return mean_sim, offset


def cross_proj(
        c1_spikes: NDArray[np.float_], 
        c2_spikes: NDArray[np.float_], 
        c1_mean: NDArray[np.float_], 
        c2_mean: NDArray[np.float_], 
        c1_norm: float, 
        c2_norm: float, 
        offset: int = 0
    ) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
    """
    Returns the cross-projections of spikes onto normalized mean waveforms for a 
    pair of clusters.
    
    Parameters
    ----------
    c1_spikes, c2_spikes: array-like
        An array of spikes in the specified cluster with shape (# of spikes, # of channels, 
        # of timepoints).
    c1_mean, c2_mean: array_like
        The mean waveform for the respective cluster. Should have shape (# of channels, # of 
        timepoints)
    c1_norm, c2_norm : array_like
       Frobenius norms for the mean waveform of the respective cluster
    offset: int
        Offset to time-shift spikes and means by before projection.
        
    Returns
    -------
    proj_1on1, proj_2on1, proj_1on2, proj_2on2: array-like
        1-D arrays containing the projections of spikes onto mean waveforms. 
    """    
    t_length: int = c1_spikes.shape[2]
    
    # adjust for offset
    if offset < 0:
        c1_spikes = c1_spikes[:,:,:t_length+offset]
        c1_mean = c1_mean[:,:t_length+offset]
        
        c2_spikes = c2_spikes[:,:,offset*-1:]
        c2_mean = c2_mean[:,offset*-1:]
    else:
        c1_spikes = c1_spikes[:,:,offset:]
        c1_mean = c1_mean[:,offset:]
    
        c2_spikes = c2_spikes[:,:,:t_length-offset]
        c2_mean = c2_mean[:,:t_length-offset]
        
    # init output arrays
    proj_1on1: NDArray[np.float_] = np.zeros((c1_spikes.shape[0]))
    proj_2on1: NDArray[np.float_] = np.zeros((c2_spikes.shape[0]))
    
    proj_1on2: NDArray[np.float_] = np.zeros((c1_spikes.shape[0]))
    proj_2on2: NDArray[np.float_] = np.zeros((c2_spikes.shape[0]))
    
    # calculate cross-projections
    norm: float = max(c1_norm, c2_norm)
    
    for i in range(c1_spikes.shape[0]):
        proj_1on1[i] = np.dot(c1_spikes[i].flatten(), c1_mean.flatten())/(c1_norm**2)
        proj_1on2[i] = np.dot(c1_spikes[i].flatten(), c2_mean.flatten())/(norm**2)
        
    for i in range(c2_spikes.shape[0]):
        proj_2on1[i] = np.dot(c2_spikes[i].flatten(), c1_mean.flatten())/(norm**2)
        proj_2on2[i] = np.dot(c2_spikes[i].flatten(), c2_mean.flatten())/(c2_norm**2)
        
    return proj_1on1, proj_2on1, proj_1on2, proj_2on2
    