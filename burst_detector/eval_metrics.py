import numpy as np
import matplotlib.pyplot as plt
import burst_detector as bd
import seaborn as sns
from scipy.stats import wasserstein_distance, pearsonr
import json
import pandas as pd
import os


def calc_fr_unif(spike_times, old2new, new2old, times_multi):
    merged_ds = np.zeros(len(new2old.keys()))
    for i in range(len(new2old.keys())):
        spike_times = []
        for clust in new2old[list(new2old.keys())[i]]:
            spike_times.append(times_multi[clust])

        spike_times = np.concatenate(spike_times)
        c1, _ = bd.bin_spike_trains(spike_times, spike_times, 20)
        n = c1.shape[0]
        merged_ds[i] = 1-wasserstein_distance(u_values=np.arange(n)/n, v_values=np.arange(n)/n, u_weights=c1, v_weights=np.ones(n))

    single_ds = np.zeros(len(old2new.keys()))
    for i in range(len(old2new.keys())):
        clust = int(list(old2new.keys())[i])
        spike_times = times_multi[clust]
        c1, _ = bd.bin_spike_trains(spike_times, spike_times, 20)
        n = c1.shape[0]
        single_ds[i] = 1-wasserstein_distance(u_values=np.arange(n)/n, v_values=np.arange(n)/n, u_weights=c1, v_weights=np.ones(n))
        
    return single_ds, merged_ds

def calc_temp_mismatch(n_clust, labels, templates, channel_map, channel_pos, mean_wf):
    # calculate template mismatch per cluster
    mismatch = np.zeros(n_clust)
    direc = np.zeros(n_clust)
    peak_channel = np.zeros(n_clust)

    for i in range(n_clust):
        if (i in counts) and (labels.loc[labels['cluster_id']==i, 'group'].item() == 'good'):
            mismatch[i] = temp_mismatch(i, templates, channel_map, channel_pos, mean_wf)

    


        
    
    















## HELPER FUNCTIONS
def get_closest_channels(channel_positions, channel_index, n=None):
    """Get the channels closest to a given channel on the probe."""
    x = channel_positions[:, 0]
    y = channel_positions[:, 1]
    x0, y0 = channel_positions[channel_index]
    d = (x - x0) ** 2 + (y - y0) ** 2
    out = np.argsort(d)
    if n:
        out = out[:n]
    return out

def get_dists(channel_positions, ref_chan, target_chan):
    x = channel_positions[:, 0]
    y = channel_positions[:, 1]
    x0, y0 = channel_positions[ref_chan]
    d = (x - x0) ** 2 + (y - y0) ** 2
    # d[y < y0] *= -1
    return d[target_chan]

def find_best_channels(template):
    amplitude_threshold = 0
    
    amplitude = template.max(axis=0) - template.min(axis=0)
    best_channel = np.argmax(amplitude)
    max_amp = amplitude[best_channel]
    
    peak_channels = np.nonzero(amplitude >= amplitude_threshold * max_amp)[0]
    
    close_channels = get_closest_channels(channel_pos, best_channel, 12)

    shank = channel_shanks[best_channel]
    channels_on_shank = np.nonzero(channel_shanks == shank)[0]
    close_channels = np.intersect1d(close_channels, channels_on_shank)
    channel_ids = np.intersect1d(close_channels, peak_channels)
    
    return channel_ids, best_channel

def temp_mismatch(clust_id, templates, channel_map, channel_pos, mean_wf):
    ch_ids, peak_channel = find_best_channels(templates[clust_id])
    
    # calculate and rank distances (proximity)
    dists = get_dists(channel_pos, peak_channel, ch_ids)
    prox_order = np.argsort(dists)
    prox_ranks = np.argsort(prox_order)
    
    # calculate and rank amplitudes
    means = mean_wf[clust_id, ch_ids, :]
    amp = means.max(axis=1) - means.min(axis=1)
    amp_order = np.argsort(amp)
    amp_ranks = np.argsort(amp_order)
    
    # calculate magnitude and direction of mismatch
    mismatch = np.abs((prox_ranks[prox_order] - amp_ranks[prox_order])[int(ch_ids.shape[0]/2):].sum()/26) # 26 is maximum possible raw mismatch
    direc = -2
    shift = channel_pos[ch_ids[amp.argmax()],1] - channel_pos[ch_ids[dists.argmin()], 1]
    
    if shift > 0:
        direc = 1
    elif shift == 0:
        direc = 0
    elif shift < 0:
        direc = -1
        
    return mismatch









