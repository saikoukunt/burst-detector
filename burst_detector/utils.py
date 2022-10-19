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
    
    return np.array(cl_times, dtype='int32')

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
        2-D list of the spike times for each of the specified clusters.
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
    
    return cl_times

def extract_spikes(data, sp_times, sp_clust, clust_id, pre_samples=20, post_samples=62, n_chan=385):
    """
    Extracts the waveforms for all spikes from the specified cluster.
    
    Parameters
    ----------
    data: array-like
        electrophysiological data of shape: (# of timepoints, # of channels).
        Should be passed in as an np.memmap for large datasets.
    sp_times: array_like
        1-D array containing all spike times.
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
        
    Returns
    -------
    spikes: array-like
        Array containing waveforms of all the spikes in the specified cluster. 
        Shape is (# of spikes, # of channels, # of timepoints) to match
        ecephys output.
    """
    times = find_times(sp_times, sp_clust, clust_id)
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