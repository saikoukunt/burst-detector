import numpy as np
from numpy.typing import NDArray

def find_times_multi(
        sp_times: NDArray[np.float_], 
        sp_clust: NDArray[np.int_], 
        clust_ids: list[int]
    )-> list[NDArray[np.float_]]:
    """
    Finds all the spike times for each of the specified clusters.
    
    Parameters
    ----------
    sp_times: array_like
        1-D array containing all spike times.
    sp_clust: array_like
        1-D array containing the cluster identity of each spike.
    clust_ids: list
        The cluster IDs of the desired spike times.
        
    Returns
    -------
    cl_times: list
        list of NumPy arrays of the spike times for each of the specified clusters.
    """
    
    # init big list and reverse dictionary
    cl_times = []
    cl2lind = {}
    for i in range(len(clust_ids)): 
        cl_times.append([])
        cl2lind[clust_ids[i]] = i 

    # count spikes in each cluster    
    for i in range(sp_times.shape[0]):
        if sp_clust[i] in cl2lind:
            cl_times[cl2lind[sp_clust[i]]].append(sp_times[i])
            
    # convert inner lists to numpy arrays
    for i in range(len(cl_times)):
        cl_times[i] = np.array(cl_times[i])
    
    return cl_times

def spikes_per_cluster(sp_clust: NDArray[np.int_])  -> dict[int, int]:
    """
    Counts the number of spikes in each cluster.
    
    Parameters
    ----------
    sp_clust: array_like
        1-D array containing the cluster identity of each spike.
        
    Returns
    -------
    spikes_per_cluster: dict
        Contains the number of spikes in each cluster, where
        the index is the cluster ID.
    
    """
    
    ids: NDArray[np.int_]; counts: NDArray[np.int_]
    ids, counts = np.unique(sp_clust, return_counts=True)
    spikes_per_cluster = dict(zip(ids, counts))
        
    return spikes_per_cluster

def extract_spikes(
        data: NDArray[np.int_], 
        times_multi: list[NDArray[np.float_]], 
        clust_id: int ,  
        pre_samples: int = 20, 
        post_samples: int = 62, 
        n_chan: int  = 385, 
        max_spikes: int = -1
    ) -> NDArray[np.int_]:
    """
    Extracts the waveforms for all spikes from the specified cluster.
    
    Parameters
    ----------
    data: array-like
        electrophysiological data of shape: (# of timepoints, # of channels).
        Should be passed in as an np.memmap for large datasets.
    times_multi: list of array_like
        List containing arrays of spike times indexed by cluster id (e.g. output of find_times_multi).
    clust_id: list
        The cluster ID of the desired spike times.
    pre_samples: int, optional
        The number of samples to extract before the peak of the spike (spike time). 
        Default matches the number in Kilosort templates.
    post_samples: int, optional
        The number of samples to extract before the peak of the spike (spike time). 
        Default matches the number in Kilosort templates.
    n_chan: int, optional
        Number of channels in the recording. Default matches Neuropixels probes.
    max_spikes: int, optional
        The maximum number of spikes to extract. Spike indices are chosen randomly 
        if max_spikes < # of spikes. If max_spikes=-1, extracts all spikes.
        
    Returns
    -------
    spikes: array-like
        Array containing waveforms of all the spikes in the specified cluster. 
        Shape is (# of spikes, # of channels, # of timepoints) to match
        ecephys output.
    """
    times: NDArray[np.int_] = times_multi[clust_id].astype("int32")
    
    # remove times that are too close to ends of recording
    while (times[0] - pre_samples) < 0:
        times = times[1:]
    while (times[-1] + post_samples) >= data.shape[0]:
        times = times[:-1]
    
    # cap number of spikes
    if (max_spikes != -1) and (max_spikes < times.shape[0]):
        np.random.shuffle(times)
        times = times[:max_spikes]
        
    # extract spikes
    spikes: NDArray[np.int_] = np.zeros((times.shape[0], n_chan, pre_samples+post_samples), dtype='int64')
    for i in range(times.shape[0]):
        spikes[i,:,:] = data[times[i]-pre_samples:times[i]+post_samples,:].T
        
    return spikes

def get_closest_channels(
        channel_positions: NDArray[np.float_], 
        channel_index: int, 
        n: int|None = None
    ) -> NDArray[np.int_]:
    """Get the channels closest to a given channel on the probe."""
    x: NDArray[np.float_] = channel_positions[:, 0]
    y: NDArray[np.float_] = channel_positions[:, 1]
    x0: float; y0: float
    x0, y0 = channel_positions[channel_index]
    d: NDArray[np.float_] = (x - x0) ** 2 + (y - y0) ** 2
    out: NDArray[np.int_] = np.argsort(d)
    if n:
        out = out[:n]
    return out

def find_best_channels(
        template: NDArray[np.float_], 
        channel_pos: NDArray[np.float_], 
        num_best: int
    ) -> tuple[NDArray[np.int_], int]:

    amplitude: NDArray[np.float_] = template.max(axis=1) - template.min(axis=1)
    best_channel: int = min(int(np.argmax(amplitude)), 382)
    
    close_channels: NDArray[np.int_] = get_closest_channels(channel_pos, best_channel, num_best)
    
    return close_channels, best_channel

def get_dists(
        channel_positions: NDArray[np.float_], 
        ref_chan: int, 
        target_chans: NDArray[np.int_]
    ) -> NDArray[np.float_]:

    x: NDArray[np.float_] = channel_positions[:, 0]
    y: NDArray[np.float_] = channel_positions[:, 1]
    x0: float; y0: float
    x0, y0 = channel_positions[ref_chan]
    d: NDArray[np.float_] = (x - x0) ** 2 + (y - y0) ** 2
    return d[target_chans]