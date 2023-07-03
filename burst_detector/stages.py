import numpy as np
import burst_detector as bd
from scipy.stats import wasserstein_distance
import pandas as pd
import math
import multiprocess as mp
from multiprocess.pool import ThreadPool as Pool
import itertools
import functools
from collections import deque

def calc_mean_sim(data, times, clusters, counts, n_clust, labels, mean_wf, params):
    
    # calculate mean waveforms if necessary
    if params['calc_means']:
        mean_wf = np.zeros(
            (n_clust, 
             params['n_chan'],
             params['pre_samples'] + params['post_samples'])
        ) 
        
        for i in range(n_clust):
            if (i in np.unique(clusters)) and (counts[i] > 0):
                spikes = bd.extract_spikes(
                    data, times, clusters, i,
                    n_chan=params['n_chan'],
                    pre_samples=params['pre_samples'],
                    post_samples=params['post_samples'],
                    max_spikes=params['max_spikes']
                )
                mean_wf[i,:,:] = np.nanmean(spikes, axis=0)
    
    # calculate mean similarity
    mean_sim, wf_norms, offset = bd.wf_means_similarity(
        mean_wf, 
        jitter=params['jitter'],
        jitter_amt=params['jitter_amt']
    )
    
    # check which cluster pairs pass threshold
    pass_ms = np.zeros_like(mean_sim, dtype='bool')
    for c1 in range(n_clust):
        for c2 in range(c1+1, n_clust):
            if (c1 in counts) and (c2 in counts):
                if mean_sim[c1,c2] >= params['sim_thresh'] and ((counts[c1] >= params['sp_num_thresh']) and (counts[c2] >= params['sp_num_thresh'])):
                    if (labels.loc[labels['cluster_id']==c1, 'group'].item() == 'good') and (labels.loc[labels['cluster_id']==c2, 'group'].item() == 'good'):
                        pass_ms[c1,c2] = True
                        pass_ms[c2,c1] = True

    return mean_sim, offset, wf_norms, mean_wf, pass_ms


def calc_cross_sim(spikes, offset, mean_wf, wf_norms, pass_ms, n_clust):
    wass_d = np.ones_like(offset, dtype='float64')
    num = 0
    
    for c1 in range(n_clust):
        for c2 in range(c1+1, n_clust):
            if pass_ms[c1,c2]:
                num += 1
                # if (num % 50 == 0):
                #     print(num)
                
                # extract spikes
                sp_1 = spikes[c1]
                sp_2 = spikes[c2]
                
                # compute cross-projections
                proj_1on1, proj_2on1, proj_1on2, proj_2on2 = bd.cross_proj(
                    sp_1,
                    sp_2,
                    mean_wf[c1],
                    mean_wf[c2],
                    wf_norms[c1],
                    wf_norms[c2],
                    offset=offset[c1, c2]
                )
                
                # bound outliers
                dist_1on1 = proj_1on1/proj_1on1
                dist_1on1[np.isnan(dist_1on1)] = 0
                
                dist_2on2 = proj_2on2/proj_2on2
                dist_2on2[np.isnan(dist_2on2)] = 0
                
                dist_1on2 = proj_1on2/proj_1on1
                dist_1on2[dist_1on2 > 1] = 1
                dist_1on2[dist_1on2 < 0] = 0
                dist_1on2[np.isnan(dist_1on2)] = 0
                
                dist_2on1 = proj_2on1/proj_2on2
                dist_2on1[dist_2on1 > 1] = 1
                dist_2on1[dist_2on1 < 0] = 0
                dist_2on1[np.isnan(dist_2on1)] = 0
                
                # compute wasserstein distances
                wass_d[c1, c2] = wasserstein_distance(dist_1on1, dist_1on2)
                wass_d[c2, c1] = wasserstein_distance(dist_2on2, dist_2on1) 
                    
    cross_sim = 1 - wass_d
    
    return cross_sim


def calc_xcorr_metric(times, clusters, n_clust, pass_ms, params):
    # define cross correlogram job
    xcorr_job = functools.partial(
        xcorr_func,
        times=times,
        clusters=clusters,
        params=params
    )
    
    # run cross correlogram jobs
    pool = mp.Pool(mp.cpu_count())
    args = []
    for c1 in range(n_clust):
        for c2 in range(c1+1, n_clust):
            if pass_ms[c1,c2]:
                args.append((c1,c2))
    res = pool.starmap(xcorr_job, args)
    
    # convert cross correlogram output to np arrays
    xgrams = np.empty_like(pass_ms, dtype='object')
    x_olaps = np.zeros_like(pass_ms, dtype='int16')
    shfl_xgrams = np.empty_like(pass_ms, dtype='object')
    shfl_olaps = np.zeros_like(pass_ms, dtype='int16')

    for i in range(len(res)):
        c1 = args[i][0]
        c2 = args[i][1]
        xgrams[c1,c2] = res[i][0]
        x_olaps[c1,c2] = res[i][1]
        # shfl_xgrams[c1,c2] = res[i][2]
        # shfl_olaps[c1,c2] = res[i][3]
        
    # compute metric
    xcorr_sig = np.zeros_like(pass_ms, dtype='float64')
    
    for c1 in range(n_clust):
        for c2 in range(c1+1, n_clust):
            if pass_ms[c1,c2]:
                xcorr_sig[c1,c2] = bd.xcorr_sig(
                    xgrams[c1,c2],
                    # shfl_xgrams[c1,c2],
                    np.ones_like(xgrams[c1,c2]),
                    window_size=params['window_size'],
                    xcorr_bin_width=params['xcorr_bin_width'],
                    max_window=params['max_window'],
                    min_xcorr_rate=params['min_xcorr_rate']
                )
    for c1 in range(n_clust):
        for c2 in range(c1+1, n_clust):
            xcorr_sig[c2,c1] = xcorr_sig[c1,c2]  
            
    return xcorr_sig, xgrams, shfl_xgrams


def calc_ref_p(times, clusters, n_clust, pass_ms, xcorr_sig, params):
    # define refractory penalty job
    ref_p_job = functools.partial(
        ref_p_func, 
        times=times, 
        clusters=clusters,
        params=params
    )
    
    # run refractory penalty job
    pool = mp.Pool(mp.cpu_count())
    args = []
    for c1 in range(n_clust):
        for c2 in range(c1+1, n_clust):
            if (pass_ms[c1,c2]) and xcorr_sig[c1,c2] > 0:
                args.append((c1,c2))

    res = pool.starmap(ref_p_job, args)
    
    # convert output to numpy arrays
    ref_pen = np.zeros_like(pass_ms, dtype='float64')
    ref_per = np.zeros_like(pass_ms, dtype='float64')
    
    for i in range(len(res)):
        c1 = args[i][0]
        c2 = args[i][1]

        ref_pen[c1,c2] = res[i][0]
        ref_pen[c2,c1] = res[i][0]
        
        ref_per[c1,c2] = res[i][1]
        ref_per[c2,c1] = res[i][1]

    return ref_pen, ref_per

def recalc_ref_p(pairs, times, clusters, counts, params):
    # define refractory penalty job
    ref_p_job = functools.partial(
        ref_p_func, 
        times=times, 
        clusters=clusters,
        params=params
    )
    
    # run refractory penalty job
    pool = mp.Pool(mp.cpu_count())
    args = list(pairs)

    res = pool.starmap(ref_p_job, args)
    
    # convert output to numpy arrays
    ref_pen = np.zeros(len(pairs), dtype='float64')
    ref_per = np.zeros(len(pairs), dtype='float64')
    
    for i in range(len(res)):
        ref_pen[i] = res[i][0]
        ref_per[i] = res[i][1]

    return ref_pen, ref_per

def merge_clusters(clusters, counts, mean_wf, final_metric, params):
    cl_max = clusters.max()
    
    # find channel with peak amplitude for each cluster
    peak_chans = np.argmax(np.max(mean_wf, 2) - np.min(mean_wf,2),1)
    
    # make rank-order list of merging
    ro_list = np.array(np.unravel_index(np.argsort(final_metric.flatten()), shape=final_metric.shape)).T[::-1][::2]
    
    # threshold list and remove pairs that are too far apart
    pairs = deque()
    ind = 0
    
    while (final_metric[ro_list[ind,0], ro_list[ind,1]] > params["final_thresh"]):
        c1_chan = peak_chans[ro_list[ind,0]] 
        c2_chan = peak_chans[ro_list[ind, 1]]
        if (np.abs(c1_chan - c2_chan) < params['max_dist']):
            pairs.append((ro_list[ind,0], ro_list[ind,1]))
        ind += 1
        
    # init dicts and lists   
    new2old = {}
    old2new = {}
    new_ind = int(clusters.max() + 1)
    
    # convert counts to array
    temp = np.zeros(cl_max)
    for i in range(cl_max):
        if i in counts:
            temp[i] = counts[i]
    counts = temp

    # merge logic
    for pair in pairs:
        c1 = int(pair[0]); c2 = int(pair[1])

        # both clusters are original
        if c1 not in old2new and c2 not in old2new:
            old2new[c1] = int(new_ind)
            old2new[c2] = int(new_ind)
            new2old[new_ind] = [c1, c2]
            new_ind += 1

        # one or more clusters has been merged already
        else:
            # get un-nested cluster list
            cl1 = new2old[old2new[c1]] if c1 in old2new else [c1]
            cl2 = new2old[old2new[c2]] if c2 in old2new else [c2]
            if cl1 == cl2:
                continue
            cl_list = cl1 + cl2
            
            # iterate through cluster pairs
            merge = True
            for i in range(len(cl_list)):
                for j in range(i+1, len(cl_list)):
                    i1 = cl_list[i]
                    i2 = cl_list[j]
                    
                    if final_metric[i1,i2] < params["final_thresh"]:
                        merge = False

            # do merge
            if merge:
                for i in range(len(cl_list)):
                    old2new[cl_list[i]] = new_ind
                new2old[new_ind] = cl_list
                new_ind += 1
                
    # remove intermediate clusters
    for key in list(set(new2old.keys())):
        if key not in list(set(old2new.values())):
            del new2old[key]
    
    return old2new, new2old


# Multithreading function definitions
# def xcorr_func(c1, c2, times, clusters, params):
#     import burst_detector as bd
    
#     # extract spike times
#     clust_times = bd.find_times_multi(times/params['fs'], clusters, [c1, c2])
#     c1_times = clust_times[0]
#     c2_times = clust_times[1]
    
#     # compute xgrams
#     return bd.calc_xgrams(
#         c1_times,
#         c2_times,
#         n_iter=params['n_iter'],
#         shuffle_bin_width=params['shuffle_bin_width'],
#         window_size=params['max_window'],
#         xcorr_bin_width=params['xcorr_bin_width'],
#         overlap_tol=params['overlap_tol']
#     )

def xcorr_func(c1, c2, times, clusters, params):
    import burst_detector as bd
    
    # extract spike times
    clust_times = bd.find_times_multi(times/params['fs'], clusters, [c1, c2])
    c1_times = clust_times[0]
    c2_times = clust_times[1]
    
    # compute xgrams
    return bd.x_correlogram(
        c1_times,
        c2_times,
        window_size=params['max_window'],
        bin_width=params['xcorr_bin_width'],
        overlap_tol=params['overlap_tol']
    )


def ref_p_func(c1, c2, times, clusters, params):
    import burst_detector as bd
    import numpy as np
    from scipy.stats import poisson
    
    # extract spike times
    cl_times = bd.find_times_multi(times/params['fs'], clusters, [c1, c2])
    c1_times = cl_times[0]
    c2_times = cl_times[1]

    # construct possible ref_ps
    ps = np.zeros(len(params["ref_pers"]))
    
    # calc autocorrelogram for merged and original clusters
    merge_times = np.sort(np.hstack((c1_times, c2_times)))
    merge_acg = bd.auto_correlogram(merge_times, bin_width=params['xcorr_bin_width'])
    
    c1_autocg = bd.auto_correlogram(c1_times, bin_width=params['xcorr_bin_width'])
    c2_autocg = bd.auto_correlogram(c2_times, bin_width=params['xcorr_bin_width'])
    
    for i in range(len(params["ref_pers"])):
        # count number of refractory period violations
        num_bins = round(params["ref_pers"][i]/params['xcorr_bin_width'])/2
        ref_start = int(int(merge_acg.shape[0]/2) - num_bins)
        ref_end = int(int(merge_acg.shape[0]/2) - 1 + num_bins)
        
        viol_1 = c1_autocg[ref_start:ref_end+1].sum()
        viol_2 = c2_autocg[ref_start:ref_end+1].sum()
        viol_merge = merge_acg[ref_start:ref_end+1].sum()
        
        viol_new = int(viol_merge - (viol_1 + viol_2))
        viol_max = int(params["max_viol"] * (len(c1_times) + len(c2_times)))
        
        if viol_new == 0:
            return 0, params["ref_pers"][0]
        
        # calc likelihood of observing max violations with Poisson rate of observed # of violations
        ps[i] = 1 - poisson.cdf(viol_max, viol_new)
    
#     if(len(c1_times) > len(c2_times)):
#         autocg = bd.auto_correlogram(c1_times, bin_width=params['xcorr_bin_width'])
#     else:
#         autocg = bd.auto_correlogram(c2_times, bin_width=params['xcorr_bin_width'])
    
#     for i in range(len(params["ref_pers"])):
#         # count number of refractory period violations
#         num_bins = round(params["ref_pers"][i]/params['xcorr_bin_width'])/2
#         ref_start = int(int(merge_acg.shape[0]/2) - num_bins)
#         ref_end = int(int(merge_acg.shape[0]/2) - 1 + num_bins)
        
#         ref_viol = int(merge_acg[ref_start:ref_end+1].sum() - autocg[ref_start:ref_end+1].sum()) # number of new collisions
#         ref_viol_max = round(params["max_viol"] * min(len(c1_times), len(c2_times)))

#         if ref_viol == 0:
#             return 0, params["ref_pers"][0]
        
#         # calc likelihood of observing max violations with Poisson rate of observed # of violations
#         ps[i] = 1 - poisson.cdf(ref_viol_max, ref_viol)
        
    return ps.min(), params["ref_pers"][ps.argmin()]