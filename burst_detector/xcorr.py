import math
import numpy as np
from numpy.typing import NDArray
import scipy
from torch import int32
import burst_detector as bd

def bin_spike_trains(
        c1_times: NDArray[np.int_], 
        c2_times: NDArray[np.int_], 
        shuffle_bin_width: float
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Splits two spike trains into bins.
    
    Parameters
    ----------
    c1_times, c2_times: array-like
        An array containing spike times in seconds.
    shuffle_bin_width: float
        Width in seconds of bins.
        
    Returns
    -------
    c1_counts, c2_counts: array-like
        An array containing binned spike counts
    """
    c1_counts: NDArray[np.float_] = np.zeros((math.ceil(max(c1_times)/shuffle_bin_width)), dtype='int32')
    c2_counts: NDArray[np.float_] = np.zeros((math.ceil(max(c2_times)/shuffle_bin_width)), dtype='int32')

    for time in c1_times:
        c1_counts[math.floor(time/shuffle_bin_width)] += 1

    for time in c2_times:
        c2_counts[math.floor(time/shuffle_bin_width)] += 1
        
    return c1_counts, c2_counts


def x_correlogram(
        c1_times: NDArray[np.float_], 
        c2_times: NDArray[np.float_], 
        window_size: float = .1, 
        bin_width: float = .001, 
        overlap_tol: float = 0
    ) -> tuple[NDArray[np.float_], int]: 
    """
    Calculates the cross correlogram between two spike trains.
    
    Parameters
    ----------
    c1_times, c2_times: array-like
        An array containing spike times (sorted least to greatest) in seconds.
    window_size: float, optional
        Width of cross correlogram window in seconds.
    bin_width: float, optional
        Width of cross correlogram bins in seconds.
    overlap_tol: float, optional
        Overlap tolerance in seconds. Spikes within the tolerance of the 
        reference spike time will not be counted for cross correlogram calculation.
        
    Returns
    -------
    corrgram: array-like
        Array containing cross correlogram with c1_times as reference spikes.
    overlap: int
        The number of overlapping spikes.
    """
    # swap so c2 is cluster with less spikes
    if c1_times.shape[0] < c2_times.shape[0]:
        temp = c1_times
        c1_times = c2_times
        c2_times = temp
    
    # init variables
    corrgram: NDArray[np.float_] = np.zeros((math.ceil(window_size/bin_width)))
    overlap = 0
    c2_start = 0
    
    # c1 are reference spikes, count c2 spikes
    for c1_ind in range((c1_times.shape[0])):
        
        # move c2 start to first spike in window
        while (c2_start < c2_times.shape[0]) and (c2_times[c2_start] < (c1_times[c1_ind] - window_size/2)):
            c2_start: int = c2_start+1
        
        # count spikes in window
        c2_ind: int = c2_start
        if(c2_ind >= c2_times.shape[0]):
            continue
        
        # update correlogram counts
        while (c2_ind < c2_times.shape[0]) and (c2_times[c2_ind] < (c1_times[c1_ind] + window_size/2)):
            
            if abs(c1_times[c1_ind] - c2_times[c2_ind]) > overlap_tol:
                gram_ind: int = min(math.floor((c1_times[c1_ind] - c2_times[c2_ind] + window_size/2)/bin_width), corrgram.shape[0]-1)
                corrgram[gram_ind] += 1
            else:
                overlap += 1
                
            c2_ind = c2_ind+1
    
    return corrgram, overlap


def auto_correlogram(
        c1_times: NDArray[np.float_], 
        window_size: float = .25, 
        bin_width: float = .001, 
        overlap_tol: float = 0
    ) -> NDArray[np.float_]: 
    """
    Calculates the auto correlogram for one spike train.
    
    Parameters
    ----------
    c1_times: array-like
        An array containing spike times (sorted least to greatest) in seconds.
    window_size: float
        Width of cross correlogram window in seconds.
    bin_width: float
        Width of cross correlogram bins in seconds.
    overlap_tol: float, optional
        Overlap tolerance in seconds. Spikes within the tolerance of the 
        reference spike time will not be counted for cross correlogram calculation.
        
    Returns
    -------
    corrgram: array-like
        Array containing autocorrelogram with c1_times.
    overlap: int
        The number of overlapping spikes.
    """
    # init variables
    corrgram: NDArray[np.float_] = np.zeros((math.ceil(window_size/bin_width)))
    start = 0
    
    # ind1 is reference spike, count ind2 spikes
    for ind1 in range(c1_times.shape[0]):
        
        # move start to first spike in window
        while (start < c1_times.shape[0]) and (c1_times[start] < (c1_times[ind1] - window_size/2)):
            start: int = start+1
        
        # count spikes in window
        ind2: int = start
        if(ind2 >= c1_times.shape[0]):
            continue
        
        while (ind2 < c1_times.shape[0]) and (c1_times[ind2] < (c1_times[ind1] + window_size/2)):
            if (ind1 != ind2) and (abs(c1_times[ind1] - c1_times[ind2]) > overlap_tol):
                gram_ind: int = min(math.floor((c1_times[ind1] - c1_times[ind2] + window_size/2)/bin_width), corrgram.shape[0]-1)
                corrgram[gram_ind] += 1
            ind2: int = ind2+1
    
    return corrgram


def xcorr_sig(
        xgram: NDArray[np.float_], 
        shfl_xgram: NDArray[np.float_], 
        window_size: float, 
        xcorr_bin_width: float, 
        max_window: float = .25, 
        min_xcorr_rate: float = 0
    ) -> float:
    """
    Calculates the cross-correlation significance metric for a cluster pair. If the ccg doesn't have a firing rate >
    min_xcorr_rate the window_size is doubled until it does or window_size == max_window. If window_size == max_window
    and the firing rate is still not high enough, the calculated sig is penalized.
    
    Parameters
    ----------
    xgram: array-like
        The raw cross-correlogram for the cluster pair.
    shfl_xgram: array_like
        The baseline cross-correlogram for the cluster pair.
    window_size: float, optional
        The width in seconds of the window used to calculate xgram and shfl_xgram.
    xcorr_bin_width: float, optional
        The width in seconds of the bins used to calculate xgram and shfl_xgram.
    min_xcorr_rate: int, optional
        The minimum number of spikes/s that must be in the cross correlogram for the
        result to be valid. Default is 0.
        
    Returns
    -------
    sig: float
        The calculated cross-correlation significance metric.
    """
    
    # reduce ccgs to window of interest
    num_bins: int = math.ceil(round(window_size/xcorr_bin_width)/2) # 0.5 of bins for whole window
    
    ref_start = int(xgram.shape[0]/2 - num_bins)
    ref_end = int(xgram.shape[0]/2 - 1 + num_bins)
    
    xgram_win: NDArray[np.float_] = xgram[ref_start:ref_end+1]
    shfl_xgram_win: NDArray[np.float_] = shfl_xgram[ref_start:ref_end+1]
    
    # expand window if xgram doesn't have a high enough 'firing' rate
    while (xgram_win.sum() < min_xcorr_rate*window_size) and (window_size < max_window):
        # expand window
        window_size = min(max_window, 2 * window_size)
        num_bins = math.ceil(round(window_size/xcorr_bin_width)/2) # 0.5 of bins for whole window
        ref_start = int(xgram.shape[0]/2 - num_bins)
        ref_end = int(xgram.shape[0]/2 - 1 + num_bins)
        xgram_win = xgram[ref_start:ref_end+1]
        shfl_xgram_win = shfl_xgram[ref_start:ref_end+1]
        
    if(xgram_win.sum() == 0) or (shfl_xgram_win.sum()==0):
        return 0
    
    # wasserstein distance
    num_bins *= 2
    sig: float = scipy.stats.wasserstein_distance(
        np.arange(num_bins)/num_bins, 
        np.arange(num_bins)/num_bins, 
        xgram_win, 
        shfl_xgram_win)/0.25  # 0.25 is maximum possible wass dist for a ccg
    
    # penalize sig for pairs with low ccg spike counts
    if min_xcorr_rate != 0:
        sig *= min(1, (xgram_win.sum()/(min_xcorr_rate * window_size))**2)
    
    return sig
