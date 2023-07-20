import numpy as np
import burst_detector as bd
import pandas as pd
import math
import os
import json
import time
import torch
from torchvision.transforms import ToTensor

def run_merge(params):
    os.makedirs(os.path.join(params['KS_folder'], "automerge"), exist_ok=True)
    os.makedirs(os.path.join(params['KS_folder'], "automerge", "spikes"), exist_ok=True)
    os.makedirs(os.path.join(params['KS_folder'], "automerge", "merges"), exist_ok=True)
    
    # load spike time and cluster estimates
    print("Loading files...")
    
    times = np.load(os.path.join(params['KS_folder'], 'spike_times.npy')).flatten()
    clusters = np.load(os.path.join(params['KS_folder'], 'spike_clusters.npy')).flatten() 
    n_clust = clusters.max() + 1
    
     # count spikes per cluster, load quality labels
    counts = bd.spikes_per_cluster(clusters)
    cl_labels = pd.read_csv(os.path.join(params['KS_folder'], 'cluster_group.tsv'), sep="\t")
    times_multi = bd.find_times_multi(times, clusters, np.arange(clusters.max()+1))
    channel_pos = np.load(os.path.join(params['KS_folder'], "channel_positions.npy"))
    
    # load recording
    rawData = np.memmap(params['data_filepath'], dtype=params['dtype'], mode='r')
    data = np.reshape(rawData, (int(rawData.size/params['n_chan']), params['n_chan']))
    
    # filter out low-spike/noise units
    cl_good = np.zeros(n_clust, dtype=bool)
    unique = np.unique(clusters)
    for i in range(n_clust):
         if (i in unique) and (counts[i] > params['min_spikes']) \
            and (cl_labels.loc[cl_labels['cluster_id']==i, 'group'].item() == 'good'):
                cl_good[i] = True
    
    
    # load mean_wf if it exists
    try:
        mean_wf = np.load(os.path.join(params['KS_folder'], 'mean_waveforms.npy'))
        std_wf = np.load(os.path.join(params['KS_folder'], 'std_waveforms.npy'))
        spikes = None
    except OSError:
        print("mean_waveforms.npy doesn't exist, calculating mean waveforms on the fly...")
        mean_wf = np.zeros(
            (n_clust, 
             params['n_chan'],
             params['pre_samples'] + params['post_samples'])
        ) 
        for i in range(n_clust):
            if cl_good[i]:
                spikes = bd.extract_spikes(
                    data, times_multi, clusters, i,
                    n_chan=params['n_chan'],
                    pre_samples=params['pre_samples'],
                    post_samples=params['post_samples'],
                    max_spikes=params['max_spikes']
                )
                mean_wf[i,:,:] = np.nanmean(spikes, axis=0)
        np.save(os.path.join(params['KS_folder'], 'mean_waveforms.npy'), mean_wf)
        np.save(os.path.join(params['KS_folder'], 'mean_waveforms.npy'), std_wf)
    peak_chans = np.argmax(np.max(mean_wf, 2) - np.min(mean_wf,2),1)
    

    t0 = time.time()
    print("Done, calculating cluster similarity...")
    
    # calculate similarity
    if params['sim_type'] == 'ae':
        # calculate nearest channels for each cluster
        chans = {}
        for i in range(mean_wf.shape[0]):
            if i in counts:
                chs, peak = bd.utils.find_best_channels(mean_wf[i], channel_pos, params['ae_chan'])
                dists = bd.utils.get_dists(channel_pos, peak, chs)
                chans[i] = chs[np.argsort(dists)].tolist()
                
        # extract spikes for ae if necessary
        if params['spikes_path'] == None:
            print("Extracting spike snippets to train autoencoder...")
            spk_fld = os.path.join(params['KS_folder'], "automerge", "spikes")
            ci = {'times': times,
                'times_multi': times_multi,
                'clusters': clusters,
                'counts': counts,
                'labels': cl_labels,
                'mean_wf': mean_wf
            }
            gti = {
                'spk_fld': spk_fld,
                'pre_samples': params['ae_pre'],
                'post_samples': params['ae_post'],
                'num_chan': params['ae_chan'],
                'noise': params['ae_noise'],
                'for_shft': params['ae_shft']
            }
            bd.generate_train_data(data, ci, channel_pos, gti, params)
            print("Spike snippets saved in " + str(spk_fld))
        else:
            spk_fld = params['spikes_path']

        # train model if necessary
        if params['model_path'] == None:
            print("Training autoencoder...")
            net, spk_data = bd.train_ae(spk_fld, counts, do_shft=params['ae_shft'], num_epochs=params['ae_epochs'])
            torch.save(net.state_dict(), os.path.join(params['KS_folder'], "automerge", "ae.pt"))
            print("Autoencoder saved in " + str(os.path.join(params['KS_folder'], "automerge", "ae.pt")))
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = bd.autoencoder.CN_AE().to(device)                         
            net.load_state_dict(torch.load(params['model_path']))
            net.eval()
            spk_data = bd.autoencoder.SpikeDataset(os.path.join(spk_fld, 'labels.csv'), spk_fld, ToTensor())
        
        # calculate ae_similarity
        sim, spk_lat_peak, lat_mean, spk_lab = bd.stages.calc_ae_sim(mean_wf, net, peak_chans, spk_data, chans, cl_good, do_shft=params['ae_shft'])
    elif params['sim_type'] == 'mean':
        # calculate mean similarity
        sim, offset, wf_norms, mean_wf, pass_ms = bd.stages.calc_mean_sim(
            data, 
            times_multi, 
            clusters, 
            counts, 
            n_clust,
            labels,
            mean_wf,
            params
        )
        sim[pass_ms == False] = 0
    pass_ms = sim > params["sim_thresh"]
    print("Found %d candidate cluster pairs" % (pass_ms.sum()/2))
    t1 = time.time()
    mean_sim_time = time.strftime('%H:%M:%S', time.gmtime(t1-t0))
    
    
    
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
    final_metric = np.zeros_like(sim)
    
    for c1 in range(n_clust):
        for c2 in range(c1, n_clust):
            met = sim[c1,c2] + \
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

    # spike_table.to_csv(os.path.join(params['KS_folder'], "automerge", "spike_table.csv"), header=False, index=False)
    # np.save(os.path.join(params['KS_folder'], "automerge", "clusters.npy"), clusters)
    # np.save(os.path.join(params['KS_folder'], "automerge", "cross_sim.npy"), stage_mets[0])
    # np.save(os.path.join(params['KS_folder'], "automerge", "xcorr_sig.npy"), stage_mets[1])
    # np.save(os.path.join(params['KS_folder'], "automerge", "ref_pen.npy"), stage_mets[2])
    if spikes == None:
        spikes = {}
        print("Caching spikes")
        for i in range(n_clust):
            if cl_good[i]:
                print("\r" + str(i) + "/" + str(clusters.max()), end="")
                spikes[i] = bd.extract_spikes(
                    data, times_multi, clusters, i,
                    n_chan=params['n_chan'],
                    pre_samples=params['pre_samples'],
                    post_samples=params['post_samples'],
                    max_spikes=params['max_spikes']
                )
            
    with open(os.path.join(params['KS_folder'], "automerge", "old2new.json"), "w") as file:
        file.write(json.dumps(old2new))

    with open(os.path.join(params['KS_folder'], "automerge", "new2old.json"), "w") as file:
        file.write(json.dumps(new2old))
        
    merges = list(new2old.values())
    bd.plot.plot_merges(merges, times_multi, mean_wf, std_wf, spikes, params)
    
    t7 = time.time()
    total_time = time.strftime('%H:%M:%S', time.gmtime(t7-t0))
    
    print("Total time: %s" % total_time)
    
    return mean_sim_time, xcorr_time, ref_pen_time, merge_time, total_time, len(list(new2old.keys())), int(clusters.max())