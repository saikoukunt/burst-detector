import numpy as np
import burst_detector as bd
import pandas as pd
import math
import os
import json
import time

def run_merge(params):
    
    # load spike time and cluster estimates
    print("Loading files...")
    
    times = np.load(os.path.join(params['KS_folder'], 'spike_times.npy')).flatten()
    clusters = np.load(os.path.join(params['KS_folder'], 'spike_clusters.npy')).flatten() 
    n_clust = clusters.max() + 1
    
    # load mean_wf if it exists
    if not params['calc_means']:
        try:
            mean_wf = np.load(os.path.join(params['KS_folder'], 'mean_waveforms.npy'))
        except OSError:
            print("mean_waveforms.npy doesn't exist, calculating mean waveforms on the fly...")
            params['calc_means'] = True
            mean_wf = None
    else:
        mean_wf = None
            
    # count spikes per cluster, load quality labels
    counts = bd.spikes_per_cluster(clusters)
    labels = pd.read_csv(os.path.join(params['KS_folder'], 'cluster_group.tsv'), sep="\t")
        
    # load recording
    rawData = np.memmap(params['data_filepath'], dtype='int16', mode='r')
    data = np.reshape(rawData, (int(rawData.size/params['n_chan']), params['n_chan']))

    t0 = time.time()
    print("Loaded data, calculating mean similarity...")
    
    # group spike times by cluster identity
    times_multi = bd.find_times_multi(times, clusters, np.arange(clusters.max()+1))
    
    # calculate mean similarity
    mean_sim, offset, wf_norms, mean_wf, pass_ms = bd.stages.calc_mean_sim(
        data, 
        times_multi, 
        clusters, 
        counts, 
        n_clust,
        labels,
        mean_wf,
        params
    )
    t1 = time.time()
    mean_sim_time = time.strftime('%H:%M:%S', time.gmtime(t1-t0))
    print("Found %d candidate cluster pairs" % (pass_ms.sum()/2))
    
    mean_sim[pass_ms == False] = 0
    
#     if not params['skip_cross_sim']:
#         print("Pre-fetching spikes for cross-projection...")
#         # pre-fetch spikes and free data RAM
#         spikes = {}
#         num = 0
#         for c1 in range(n_clust):
#             for c2 in range(c1+1, n_clust):
#                 if pass_ms[c1,c2]:
#                     num += 1

#                     if c1 not in spikes:
#                         spikes[c1] = bd.extract_spikes(
#                             data, 
#                             times_multi, 
#                             clusters, 
#                             c1, 
#                             pre_samples=params['pre_samples'],
#                             post_samples=params['post_samples'],
#                             n_chan=params['n_chan'],
#                             max_spikes=params['max_spikes']
#                         )
#                     if c2 not in spikes:
#                         spikes[c2] = bd.extract_spikes(
#                             data, 
#                             times_multi, 
#                             clusters, 
#                             c2, 
#                             pre_samples=params['pre_samples'],
#                             post_samples=params['post_samples'],
#                             n_chan=params['n_chan'],
#                             max_spikes=params['max_spikes']
#                         )
#         data = None

#         t2 = time.time()
#         cache_spikes_time = time.strftime('%H:%M:%S', time.gmtime(t2-t1))
#         print("Caching spikes took %s" % cache_spikes_time)
#         print("Calculating cross-projection similarity...")
        
#         # calculate cross-projection similarity and free spikes RAM
#         cross_sim = bd.stages.calc_cross_sim(spikes, offset, mean_wf, wf_norms, pass_ms, n_clust)
#         spikes = None
#         t3 = time.time()
#         cross_sim_time = time.strftime('%H:%M:%S', time.gmtime(t3-t2))
#         print("Cross projection took %s" % cross_sim_time)
    
    # calculate xcorr metric
    print("Calculating cross-correlation metric...")
    xcorr_sig, xgrams, shfl_xgrams = bd.stages.calc_xcorr_metric(
        times, 
        clusters,
        n_clust, 
        pass_ms, 
        params
    )
    t4 = time.time()
    xcorr_time = time.strftime('%H:%M:%S', time.gmtime(t4-t1))
    print("Cross correlation took %s" % xcorr_time)
    
    # calculate refractory penalty
    print("Calculating refractory period penalty...")
    ref_pen, ref_per = bd.stages.calc_ref_p(times, clusters, n_clust, pass_ms, xcorr_sig,\
                                   params)
    t5 = time.time()
    ref_pen_time = time.strftime('%H:%M:%S', time.gmtime(t5-t4))
    print("Refractory period penalty took %s" % ref_pen_time)

    # calculate final metric
    print("Calculating final metric...")
    final_metric = np.zeros_like(mean_sim)
    
    for c1 in range(n_clust):
        for c2 in range(c1, n_clust):
            met = mean_sim[c1,c2] + \
            params['xcorr_coeff']*xcorr_sig[c1,c2] - \
            params['ref_pen_coeff']*ref_pen[c1,c2]
            
            final_metric[c1,c2] = max(met, 0)
            final_metric[c2,c1] = max(met, 0)
            
            
    print("Merging...")
    
    channel_map = np.load(os.path.join(params['KS_folder'], 'channel_map.npy')).flatten()
    
    old2new, new2old = bd.stages.merge_clusters(
        clusters, counts, mean_wf, final_metric, params)
   
    t6 = time.time()
    merge_time = time.strftime('%H:%M:%S', time.gmtime(t6-t5))
    
    print("Merging took %s" % merge_time)
    
    print("Writing to output...")
    # write spike table output
    # weighted avg new waveforms
#     new2old = merge_lists[1]

#     old_n = mean_wf.shape[0]
#     n = clusters.max() + 1
#     temp = np.zeros((n, mean_wf.shape[1], mean_wf.shape[2]))
#     temp[:old_n,:,:] = mean_wf
#     mean_wf = temp
    
#     # convert counts to array
#     temp = np.zeros(n_clust)
#     for i in range(n_clust):
#         if i in counts:
#             temp[i] = counts[i]
#     counts = temp
    
#     temp = np.zeros(n)
#     temp[:old_n] = counts
#     counts = temp

            
#     # calculate peak channels 
#     mean_wf = mean_wf[:, channel_map, :]
#     peak_chans = np.argmax(np.max(mean_wf, 2) - np.min(mean_wf,2),1)
#     channel_pos = np.load(os.path.join(params['KS_folder'], 'channel_positions.npy'))

#     # construct spike table dataframe
#     X = channel_pos[peak_chans[clusters],0]
#     Y = channel_pos[peak_chans[clusters],1]
    
#     spike_table = pd.DataFrame(
#         {"time": times,
#         "cluster": clusters,
#         "X": X,
#         "Y": Y}
    # )

    os.makedirs(os.path.join(params['KS_folder'], "automerge"), exist_ok=True)
    # spike_table.to_csv(os.path.join(params['KS_folder'], "automerge", "spike_table.csv"), header=False, index=False)
    # np.save(os.path.join(params['KS_folder'], "automerge", "clusters.npy"), clusters)
    # np.save(os.path.join(params['KS_folder'], "automerge", "cross_sim.npy"), stage_mets[0])
    # np.save(os.path.join(params['KS_folder'], "automerge", "xcorr_sig.npy"), stage_mets[1])
    # np.save(os.path.join(params['KS_folder'], "automerge", "ref_pen.npy"), stage_mets[2])

    with open(os.path.join(params['KS_folder'], "automerge", "old2new.json"), "w") as file:
        file.write(json.dumps(old2new))

    with open(os.path.join(params['KS_folder'], "automerge", "new2old.json"), "w") as file:
        file.write(json.dumps(new2old))
        
    merges = list(new2old.values())
    bd.plot.plot_merges(merges, times_multi, mean_wf, params)
    
    t7 = time.time()
    total_time = time.strftime('%H:%M:%S', time.gmtime(t7-t0))
    
    print("Total time: %s" % total_time)
    
    
    return mean_sim_time, xcorr_time, ref_pen_time, merge_time, total_time, len(list(new2old.keys())), int(clusters.max())