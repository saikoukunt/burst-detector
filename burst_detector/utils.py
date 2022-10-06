import numpy as np

def find_times(sp_times, sp_clust, clust_ids):
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