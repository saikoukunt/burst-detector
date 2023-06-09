import math
import numpy as np
import scipy
import burst_detector as bd

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


def x_correlogram(c1_times, c2_times, window_size=.1, bin_width=.001, overlap_tol=0): 
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
    corrgram = np.zeros((math.ceil(window_size/bin_width)))
    overlap = 0
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
        
        # update correlogram counts
        while (c2_ind < c2_times.shape[0]) and (c2_times[c2_ind] < (c1_times[c1_ind] + window_size/2)):
            
            if abs(c1_times[c1_ind] - c2_times[c2_ind]) > overlap_tol:
                gram_ind = min(math.floor((c1_times[c1_ind] - c2_times[c2_ind] + window_size/2)/bin_width), corrgram.shape[0]-1)
                corrgram[gram_ind] += 1
            else:
                overlap += 1
                
            c2_ind = c2_ind+1
    
    return corrgram, overlap


def auto_correlogram(c1_times, window_size=.25, bin_width=.001, overlap_tol=0): 
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
    corrgram = np.zeros((math.ceil(window_size/bin_width)))
    overlap = 0
    start = 0
    
    # ind1 is reference spike, count ind2 spikes
    for ind1 in range(c1_times.shape[0]):
        
        # move start to first spike in window
        while (start < c1_times.shape[0]) and (c1_times[start] < (c1_times[ind1] - window_size/2)):
            start = start+1
        
        # count spikes in window
        ind2 = start
        if(ind2 >= c1_times.shape[0]):
            continue
        
        while (ind2 < c1_times.shape[0]) and (c1_times[ind2] < (c1_times[ind1] + window_size/2)):
            if (ind1 != ind2) and (abs(c1_times[ind1] - c1_times[ind2]) > overlap_tol):
                gram_ind = min(math.floor((c1_times[ind1] - c1_times[ind2] + window_size/2)/bin_width), corrgram.shape[0]-1)
                corrgram[gram_ind] += 1
            ind2 = ind2+1
    
    return corrgram


def xcorr_sig(xgram, shfl_xgram, window_size, xcorr_bin_width, max_window=.25, min_xcorr_rate=0):
    """
    Calculates the cross-correlation significance metric for a cluster pair. Returns 0 if the cross correlogram
    does not contain more than min_xcorr_spikes spikes.
    
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
    num_bins = math.ceil(round(window_size/xcorr_bin_width)/2) # 0.5 of bins for whole window
    
    ref_start = int(xgram.shape[0]/2 - num_bins)
    ref_end = int(xgram.shape[0]/2 - 1 + num_bins)
    
    xgram_win = xgram[ref_start:ref_end+1]
    shfl_xgram_win = shfl_xgram[ref_start:ref_end+1]
    
    # check to make sure xgram has enough spikes
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
    sig = scipy.stats.wasserstein_distance(np.arange(num_bins)/num_bins, np.arange(num_bins)/num_bins, xgram_win, shfl_xgram_win)/0.25  # 0.25 is max in this context
    
    # penalize sig for pairs with low ccg spike counts
    if min_xcorr_rate != 0:
        sig *= min(1, (xgram_win.sum()/(min_xcorr_rate * window_size))**2)
    
    return sig


def calc_xgrams(c1_times, c2_times, n_iter=50, shuffle_bin_width=.1, window_size=.25, xcorr_bin_width=.001, overlap_tol=0):
    """
    Calculates the raw and baseline cross-correlations between two clusters.
    
    Parameters
    ----------
    c1_times, c2_times: array-like
        An array containing spike times in seconds.
    n_iter: int, optional
        The number of shuffle iterations for the baseline cross-correlation calculation. 
        Default value is 200.
    shuffle_bin_width: float, optional
        The width of bins in seconds for spike train shuffling. Default value is .1.
    window_size: float, optional
        The width in seconds of the cross correlogram window. Default value is .25.
    xcorr_bin_width: float, optional
        The width in seconds of the bins for cross correlogram calculation. Default value 
        is .001.
    overlap_tol: float, optional
        Overlap tolerance in seconds. Spikes within the tolerance of the 
        reference spike time will not be counted for cross correlogram calculation.

    Returns
    -------
    xgram: array-like
        The actual cross-correlogram for the cluster pair.
    shfl_xgram: array_like
        The baseline cross-correlogram for the cluster pair, calculated by averaging
        cross-correlograms for shuffled versions of the spike trains.
    xgram_overlap, shfl_overlap: float
        The number of overlapping spikes for the respective correlograms.
    """
     
    # bin spike trains for shuffling
    c1_counts, c2_counts = bin_spike_trains(c1_times, c2_times, shuffle_bin_width)

    # init
    n_bins = math.ceil(window_size/xcorr_bin_width)
    shfl_xgram = np.zeros(n_bins)
    shfl_overlap = 0

    # do shuffled iterations
    for i in range(n_iter):
        c1_shfl, c2_shfl = shuffle_spike_trains(c1_counts, c2_counts, shuffle_bin_width)
        
        shfl, olap = x_correlogram(c1_shfl, c2_shfl, window_size, xcorr_bin_width, overlap_tol=overlap_tol)
        shfl_xgram += shfl/n_iter
        shfl_overlap += olap/n_iter

    xgram, xgram_overlap = x_correlogram(c1_times, c2_times, window_size, xcorr_bin_width, overlap_tol=overlap_tol)

    return xgram, xgram_overlap, shfl_xgram, shfl_overlap


def get_isi_est(sp_times, sp_clust, counts, sp_num_thresh, fs=30000, isi_prct=10, isi_est_max = .25, isi_est_min = .025):
    """
    Finds an ISI percentile based estimate of the cross-correlation window for all clusters in a recording.
    
    Parameters
    ----------
    sp_times: array_like
        1-D array containing all spike times.
    sp_clust: array_like
        1-D array containing the cluster identity of each spike.
    counts: array_like
        1-D array containing the number of spikes in each cluster.
    sp_num_thresh: int
        Minimum number of spikes that a cluster must have to be considered.
    fs: int, optional
        Sampling frequency of the recording in Hz. Default value is 30000.
    isi_prct: float
        ISI percentile to use to estimate refractory period. Default value is 0.1.
    isi_est_max, isi_est_min: float
        Bounds for refractory period estimates in seconds. Default bounds are [.0005, .05]
        
    Returns
    -------
    isi_est: array_like
        Array of refractory period estimates, indexed by cluster ID.
    """
   # calculate refractory period for all clusters
    isi_est = np.zeros(counts.shape[0])

    all_times = bd.find_times_multi(sp_times/fs, sp_clust, range(counts.shape[0]))

    for i in range(counts.shape[0]):
        if counts[i] >= sp_num_thresh:
            isi_est[i] = np.percentile(np.diff(all_times[i]), isi_prct)

    # bound result
    isi_est[isi_est < isi_est_min] = isi_est_min
    isi_est[isi_est > isi_est_max] = isi_est_max
    
    # insert 0 for small clusters
    for i in range(counts.shape[0]):
        if counts[i] < sp_num_thresh:
            isi_est[i] = 0
    
    return isi_est


def calc_ref_pen(x, start, stop):
    """
    Applies a clipped ReLu function to the input value or vector.
    
    Parameters
    ----------
    x: array-like or float
        Input value(s) to the sigmoid function.
    start: float
        x-value where the sigmoid function should start to rise rapidly.
    stop: float
        x-value where the sigmoid function should finish rising rapidly.
        
    Returns
    -------
    ref_pen: array-like or float
        Output value(s) of the clipped ReLu function.
    """
    
    if(stop <= start):
        return 0
    
    # calculate penalty
    ref_pen =  (x-start)/(stop-start)
    
    # bound penalty
    ref_pen = min(max(ref_pen,0),1)

    return ref_pen
