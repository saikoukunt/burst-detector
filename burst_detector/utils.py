import numpy as np

def find_times(sp_times, sp_clust, clust_id):
    """
    Finds all the spike times for the specified cluster.
    
    Parameters
    ----------
    sp_times: array_like
        1-D array containing all spike times.
    sp_clust: array_like
        1-D array containing the cluster identity of each spike.
    clust_id: list
        The cluster ID of the desired spike times.
        
    Returns
    -------
    cl_times: array_like
        1-D array of the spike times for the specified cluster.
    """
    
    cl_times = []

    # count spikes in each cluster    
    for i in np.arange(sp_times.shape[0]):
        if sp_clust[i] == clust_id:
            cl_times.append(sp_times[i])
    
    return np.array(cl_times)

def find_times_multi(sp_times, sp_clust, clust_ids):
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
    for i in np.arange(len(clust_ids)): 
        cl_times.append([])
        cl2lind[clust_ids[i]] = i 

    # count spikes in each cluster    
    for i in np.arange(sp_times.shape[0]):
        if sp_clust[i] in cl2lind:
            cl_times[cl2lind[sp_clust[i]]].append(sp_times[i])
            
    # convert inner lists to numpy arrays
    for i in range(len(cl_times)):
        cl_times[i] = np.array(cl_times[i])
    
    return cl_times

def spikes_per_cluster(sp_clust):
    """
    Counts the number of spikes in each cluster.
    
    Parameters
    ----------
    sp_clust: array_like
        1-D array containing the cluster identity of each spike.
        
    Returns
    -------
    counts: array_like
        1-D array containng the number of spikes in each cluster, where
        the index is the cluster ID.
    
    """
    
    ids, counts = np.unique(sp_clust, return_counts=True)
    counts = dict(zip(ids, counts))
    
#     counts = np.zeros((sp_clust.max()+1), dtype='uint16')
    
#     for clust_id in sp_clust:
#         counts[clust_id] += 1
        
    return counts

def extract_spikes(data, times_multi, sp_clust, clust_id, pre_samples=20, post_samples=62, n_chan=385, max_spikes=-1):
    """
    Extracts the waveforms for all spikes from the specified cluster.
    
    Parameters
    ----------
    data: array-like
        electrophysiological data of shape: (# of timepoints, # of channels).
        Should be passed in as an np.memmap for large datasets.
    times_multi: list of array_like
        List containing arrays of spike times indexed by cluster id (e.g. output of find_times_multi).
    sp_clust: array_like
        1-D array containing the cluster identity of each spike.
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
    times = times_multi[clust_id].astype("int32")
    
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
    spikes = np.zeros((times.shape[0], n_chan, pre_samples+post_samples))
    for i in range(times.shape[0]):
        spikes[i,:,:] = data[times[i]-pre_samples:times[i]+post_samples,:].T
        
    return spikes
    
    
def fix_clust_ids(sp_clust, n_clusters):
    """
    Fixes cluster numbering so that cluster IDs are continuous.
    
    Parameters
    ----------
    sp_clust: array_like
        1-D array containing the cluster identity of each spike.
    n_clusters: int
        The number of clusters.
        
    Returns 
    -------
    clusters: array_like
        1-D array containing the fixed cluster IDs.
    old2new:
        Dictionary from old cluster IDs to the fixed ones.
    """
    
    # build dict
    old2new = {}
    clust_ids = np.unique(sp_clust)
    for i in np.arange(n_clusters):
        old2new[clust_ids[i]] = i;
       
    # fix cluster ids
    n = sp_clust.shape[0]
    clusters = np.zeros(n, dtype=np.uint)
    for i in np.arange(n):
        clusters[i] = old2new[sp_clust[i]]
        
    return clusters, old2new
    
def clust_counts(sp_clust, n_clusters):
    """
    Counts the number of spikes in each cluster.
    
    Parameters
    ----------
    sp_clust: array_like
        1-D array containing the cluster identity of each spike. Cluster numbering 
        must be continuous (like the output of fix_clust_ids).
    n_clusters: int
        The number of clusters.

    Returns
    -------
    counts: array_like
        1-D array containing the number of spikes in each cluster.
    """

    # count number of spikes for each cluster
    counts = np.zeros(n_clusters)
    n = sp_clust.shape[0]
    for i in np.arange(n):
        counts[sp_clust[i]] += 1
    
    return counts

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

def find_best_channels(template, channel_pos, num_best):
    amplitude_threshold = 0
    
    amplitude = template.max(axis=1) - template.min(axis=1)
    best_channel = min(np.argmax(amplitude), 382)
    max_amp = amplitude[best_channel]
    
    peak_channels = np.nonzero(amplitude >= amplitude_threshold * max_amp)[0]
    close_channels = get_closest_channels(channel_pos, best_channel, num_best)

    # channel_shanks = (channel_pos[:,0]/250).astype("int")
    # shank = channel_shanks[best_channel]
    # channels_on_shank = np.nonzero(channel_shanks == shank)[0]
    # close_channels = np.intersect1d(close_channels, channels_on_shank)
    # channel_ids = np.intersect1d(close_channels, peak_channels)
    
    return close_channels, best_channel

def get_dists(channel_positions, ref_chan, target_chan):
    x = channel_positions[:, 0]
    y = channel_positions[:, 1]
    x0, y0 = channel_positions[ref_chan]
    d = (x - x0) ** 2 + (y - y0) ** 2
    # d[y < y0] *= -1
    return d[target_chan]