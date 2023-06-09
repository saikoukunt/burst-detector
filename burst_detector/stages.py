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
    mean_sim, offset, wf_means_norm = bd.wf_means_similarity(
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

    return mean_sim, offset, wf_means_norm, mean_wf, pass_ms


def calc_cross_sim(spikes, wf_means_norm, offset, pass_ms, n_clust):
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
                    wf_means_norm[c1],
                    wf_means_norm[c2],
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

def merge_clusters(times, clusters, counts, n_clust, final_metric, cross_sim, xcorr_sig, ref_pen, mean_wf, channel_map, params):
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
    rejected = []
    orig_cl = cross_sim.shape[0] 
    
    # convert counts to array
    temp = np.zeros(n_clust)
    for i in range(n_clust):
        if i in counts:
            temp[i] = counts[i]
    counts = temp

    # merge logic
    while pairs:
        n = len(pairs)
        for i in range(n):
            pair = pairs.popleft(); c1 = int(pair[0]); c2 = int(pair[1])
            
            if (c1 not in old2new) and (c2 not in old2new): # check if one of clusters has already been merged
                # threshold metric if needed
                if (c1 < orig_cl) and (c2 < orig_cl):
                    merge = True
                else:
                    metric = np.sqrt(cross_sim[c1,c2]*cross_sim[c2,c1]) + params["xcorr_coeff"]*xcorr_sig[c1,c2] \
                    - params["ref_pen_coeff"]*ref_pen[c1,c2]
                    merge = metric > params['final_thresh']
                # perform merge
                if merge:
                    old2new[c1] = new_ind
                    old2new[c2] = new_ind
                    new2old[new_ind] = (c1, c2)
                    new_ind += 1
                else:
                    rejected.append((c1,c2))
            else:
                # convert merged clusters ids to new ones
                if c1 in old2new:
                    c1 = old2new[c1]
                if c2 in old2new:
                    c2 = old2new[c2]
                if c1 != c2:
                    pairs.append((c1,c2))
                    
        # edge case id conversion
        for i in range(len(pairs)):
            c1 = pairs[i][0]
            c2 = pairs[i][1]
            if c1 in old2new:
                c1 = old2new[c1]
            if c2 in old2new:
                c2 = old2new[c2]
            pairs[i] = (c1,c2)
            
            
        # update cross_sim with weighted avg
        nc = counts / counts.max() # prescale to prevent overflow
        temp = np.zeros((new_ind, new_ind))
        old_n = cross_sim.shape[0]
        temp[:old_n, :old_n] = cross_sim
        cross_sim = temp
        for i in range(old_n, new_ind): # new-old intersections
            c1 = new2old[i][0]
            c2 = new2old[i][1]
            cross_sim[i,:old_n] = (nc[c1]*cross_sim[c1,:old_n] + nc[c2]*cross_sim[c2,:old_n])/(nc[c1]+nc[c2])
            cross_sim[:old_n, i] = (nc[c1]*cross_sim[:old_n,c1] + nc[c2]*cross_sim[:old_n,c2])/(nc[c1]+nc[c2])
            
        for i in range(old_n, new_ind): # new-new intersections
            c1 = new2old[i][0]
            c2 = new2old[i][1]
            for j in range(old_n+1, new_ind):
                c3 = new2old[j][0]
                c4 = new2old[j][1]
                cross_sim[i,j] = nc[c1]*nc[c3]*cross_sim[c1,c3] + nc[c1]*nc[c4]*cross_sim[c1,c4] + \
                nc[c2]*nc[c3]*cross_sim[c2,c3] + nc[c2]*nc[c4]*cross_sim[c2,c4]
                cross_sim[i,j] /= ((nc[c1]+nc[c2])*(nc[c3]+nc[c4]))
                
       # update xcorr_sig with weighted avges
        temp = np.zeros((new_ind, new_ind))
        old_n = xcorr_sig.shape[0]
        temp[:old_n, :old_n] = xcorr_sig
        xcorr_sig = temp
        for i in range(old_n, new_ind): # new-old intersections
            c1 = new2old[i][0]
            c2 = new2old[i][1]
            xcorr_sig[i,:old_n] = (nc[c1]*xcorr_sig[c1,:old_n] + nc[c2]*xcorr_sig[c2,:old_n])/(nc[c1]+nc[c2])
            xcorr_sig[:old_n, i] = (nc[c1]*xcorr_sig[:old_n,c1] + nc[c2]*xcorr_sig[:old_n,c2])/(nc[c1]+nc[c2])
            
        for i in range(old_n, new_ind): # new-new intersections
            c1 = new2old[i][0]
            c2 = new2old[i][1]
            for j in range(old_n+1, new_ind):
                c3 = new2old[j][0]
                c4 = new2old[j][1]
                xcorr_sig[i,j] = nc[c1]*nc[c3]*xcorr_sig[c1,c3] + nc[c1]*nc[c4]*xcorr_sig[c1,c4] + \
                nc[c2]*nc[c3]*xcorr_sig[c2,c3] + nc[c2]*nc[c4]*xcorr_sig[c2,c4]
                xcorr_sig[i,j] /= ((nc[c1]+nc[c2])*(nc[c3]+nc[c4]))
                
        # update clusters, counts
        for i in range(clusters.shape[0]):
            if clusters[i] in old2new:
                clusters[i] = old2new[clusters[i]]
        
        counts = bd.spikes_per_cluster(clusters)
        n_clust = clusters.max() + 1
        # convert counts to array
        temp = np.zeros(n_clust)
        for i in range(n_clust):
            if i in counts:
                temp[i] = counts[i]
        counts = temp
        
        # recalculate ref_pen
        rp, rr = recalc_ref_p(pairs, times, clusters, counts, params)
        temp = np.zeros((new_ind, new_ind))
        old_n = ref_pen.shape[0]
        temp[:old_n, :old_n] = ref_pen
        ref_pen = temp
        
        for i in range(len(pairs)):
            ref_pen[pairs[i][0], pairs[i][1]] = rp[i]
        
    # build lists to return
    stage_mets = [cross_sim, xcorr_sig, ref_pen]
    merge_lists = [old2new, new2old, rejected]
    
    return clusters, counts, merge_lists, stage_mets


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