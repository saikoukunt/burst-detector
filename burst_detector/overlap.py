import numpy as np
from collections import deque
import burst_detector as bd

def overlap(sp_times, sp_clust, n_clusters, fs):
    """
    Calculates the number of spikes that overlap (i.e. spike times within 2 ms) for each 
    pair of clusters.
    
    Parameters
    ----------
    sp_times: array_like
        1-D array containing all spike times.
    sp_clust: array_like
        1-D array containing the cluster identity of each spike. Cluster numbering 
        must be continuous (like the output of fix_clust_ids).
    n_clusters: int
        The number of clusters.
    fs: int
        Sampling rate of the recording in Hz.
        
    Returns
    -------
    olap: array_like
        A symmetrical 2-D array containing the number of spikes that overlap for each pair of clusters.
    """
    
    # count overlapping spikes
    tmplt_len = 61/fs
    window = deque()
    window_counts = np.zeros(n_clusters)
    olap = np.zeros((n_clusters, n_clusters))

    n = sp_times.shape[0]
    for i in np.arange(n):
        window.append(i)
        # Boot spikes outside window
        while (sp_times[i] - sp_times[window[0]]) > tmplt_len:
            old = window.popleft()
            window_counts[sp_clust[old]] -= 1

        # update overlap row of new spike
        olap[sp_clust[i], :] += window_counts

        # update overlap rows for pre-existing spikes
        olap[:, sp_clust[i]] += window_counts.T

        # update window counts
        window_counts[sp_clust[i]] += 1 
        
    return olap
        
def overlap_norm(sp_times, sp_clust, n_clusters, fs):
    """
    Calculates the proportion of spikes that overlap (i.e. spike times within 2 ms) for each 
    pair of clusters.
    
     Parameters
    ----------
    sp_times: array_like
        1-D array containing all spike times.
    sp_clust: array_like
        1-D array containing the cluster identity of each spike. Cluster numbering 
        must be continuous (like the output of fix_clust_ids).
    n_clusters: int
        The number of clusters.
    fs: int
        Sampling rate of the recording in Hz.
        
    Returns
    -------
    olap_norm: array_like
        A 2-D array containing the proportion of spikes that overlap for each pair of clusters.
        Each row is calculated by normalizing the overlap counts (from overlap()) by the total
        number of spikes in the row cluster.
    
    """
    
    olap_norm = overlap(sp_times, sp_clust, n_clusters, fs)
    cc = bd.clust_counts(sp_clust, n_clusters)
    
    for i in np.arange(n_clusters):
        olap_norm[i, :] /= cc[i]
        
    return olap_norm
    
