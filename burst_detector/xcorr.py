import math
import numpy as np
import scipy

def bin_spike_trains(c1_times, c2_times, shuffle_bin_width):
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
    
    c1_counts = np.zeros((math.ceil(max(c1_times)/shuffle_bin_width)), dtype='int32')
    c2_counts = np.zeros((math.ceil(max(c2_times)/shuffle_bin_width)), dtype='int32')

    for time in c1_times:
        c1_counts[math.floor(time/shuffle_bin_width)] += 1

    for time in c2_times:
        c2_counts[math.floor(time/shuffle_bin_width)] += 1
        
    return c1_counts, c2_counts


def shuffle_spike_trains(c1_counts, c2_counts, shuffle_bin_width):
    """
    Randomizes spike times in two spike trains. Spike trains are split into bins and the 
    spike times in each bin are randomized while preserving the number of spikes per bin.
    
    Parameters
    ----------
    c1_counts, c2_counts: array-like
        An array containing binned spike counts, like the output of bin_spike_trains.
    shuffle_bin_width: float
        Width in seconds of bins to shuffle within.
        
    Returns
    -------
    c1_shfl, c2_shfl: array-like
        An array containing shuffled spike times.
    """
    
    c1_shfl = np.zeros(int(c1_counts.sum()))
    c2_shfl = np.zeros(int(c2_counts.sum()))

    c1_ind = 0
    c2_ind = 0

    for i in range(c1_counts.shape[0]):
        if c1_counts[i] > 0:
            spks = np.cumsum(np.random.exponential(shuffle_bin_width/c1_counts[i] ,c1_counts[i]))
            c1_shfl[c1_ind:c1_ind+c1_counts[i]] = spks + i*shuffle_bin_width
            c1_ind += c1_counts[i]

    for i in range(c2_counts.shape[0]):        
        if c2_counts[i] > 0:
            spks = np.cumsum(np.random.exponential(shuffle_bin_width/c2_counts[i] ,c2_counts[i]))
            c2_shfl[c2_ind:c2_ind+c2_counts[i]] = spks + i*shuffle_bin_width
            c2_ind += c2_counts[i]
            
    return np.sort(c1_shfl), np.sort(c2_shfl)


def x_correlogram(c1_times, c2_times, window_size=.1, bin_width=.001): 
    """
    Calculates the cross correlogram between two spike trains.
    
    Parameters
    ----------
    c1_times, c2_times: array-like
        An array containing spike times in seconds.
    window_size: float
        Width of cross correlogram window in seconds.
    bin_width: float
        Width of cross correlogram bins in seconds.
        
    Returns
    -------
    corrgram: array-like
        Array containing cross correlogram with c1_times as reference spikes.
    """
    # swap so c2 is cluster with less spikes
    if c1_times.shape[0] < c2_times.shape[0]:
        temp = c1_times
        c1_times = c2_times
        c2_times = temp
    
    # init variables
    corrgram = np.zeros((math.ceil(window_size/bin_width)))
    c2_start = 0
    
    # c1 are reference spikes, count c2 spikes
    for c1_ind in range((c1_times.shape[0])):
        
        # move c2 start to first spike in window
        while (c2_start < c2_times.shape[0]) and (c2_times[c2_start] < (c1_times[c1_ind] - window_size/2)):
            c2_start = c2_start+1
        
        # count spikes in window
        c2_ind = c2_start
        if(c2_ind >= c2_times.shape[0]):
            continue
        
        while (c2_ind < c2_times.shape[0]) and (c2_times[c2_ind] < (c1_times[c1_ind] + window_size/2)):
            gram_ind = min(math.floor((c1_times[c1_ind] - c2_times[c2_ind] + window_size/2)/bin_width), corrgram.shape[0]-1)
            corrgram[gram_ind] += 1
            c2_ind = c2_ind+1
    
    return corrgram

def xcorr_sig(c1_times, c2_times, n_iter=200, shuffle_bin_width=.1, window_size=.1, xcorr_bin_width=.001):
    
    # bin spike trains for shuffling
    c1_counts, c2_counts = bin_spike_trains(c1_times, c2_times, shuffle_bin_width)
    
    # do shuffled iterations
    n_bins = math.ceil(window_size/xcorr_bin_width)
    shfl_xgram = np.zeros(n_bins)
    
    for i in range(n_iter):
        c1_shfl, c2_shfl = shuffle_spike_trains(c1_counts, c2_counts, shuffle_bin_width)
        shfl_xgram += x_correlogram(c1_shfl, c2_shfl, window_size, xcorr_bin_width)/n_iter
    
    xgram = x_correlogram(c1_times, c2_times, window_size, xcorr_bin_width)
    
    # normalize so we only compare shape
    xgram /= xgram.sum()
    shfl_xgram /= shfl_xgram.sum()
    
    # wasserstein distance
    sig = scipy.stats.wasserstein_distance(np.arange(n_bins)/n_bins, np.arange(n_bins)/n_bins, xgram, shfl_xgram)/0.25  # 0.25 is max in this context
    
    return sig, xgram, shfl_xgram