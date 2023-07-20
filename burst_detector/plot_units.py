import burst_detector as bd
from argschema import ArgSchemaParser
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import math

def main():
    from burst_detector import AutoMergeParams
    
    mod = ArgSchemaParser(schema_type=AutoMergeParams)
    params = mod.args

    os.makedirs(os.path.join(params['KS_folder'], "automerge"), exist_ok=True)
    os.makedirs(os.path.join(params['KS_folder'], "automerge", "units"), exist_ok=True)
    
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
         if (i in unique) and (counts[i] > params['min_spikes']):
                cl_good[i] = True
    try:
        mean_wf = np.load(os.path.join(params['KS_folder'], 'mean_waveforms.npy'))
        std_wf = np.load(os.path.join(params['KS_folder'], 'std_waveforms.npy'))
        
        print("Caching spikes...")
        spikes = {}
        for i in range(n_clust):
            print("\r" + str(i) + "/" + str(clusters.max()), end="")
            if cl_good[i]:
                spikes[i] = bd.extract_spikes(
                    data, times_multi, clusters, i,
                    n_chan=params['n_chan'],
                    pre_samples=params['pre_samples'],
                    post_samples=params['post_samples'],
                    max_spikes=params['max_spikes']
                )
        
    except OSError:
        print("mean_waveforms.npy doesn't exist, calculating mean waveforms on the fly...")
        mean_wf = np.zeros(
            (n_clust, 
             params['n_chan'],
             params['pre_samples'] + params['post_samples'])
        ) 
        std_wf = np.zeros_like(mean_wf)
        spikes = {}
        
        for i in range(n_clust):
            print("\r" + str(i) + "/" + str(clusters.max()), end="")
            if cl_good[i]:
                spikes[i] = bd.extract_spikes(
                    data, times_multi, clusters, i,
                    n_chan=params['n_chan'],
                    pre_samples=params['pre_samples'],
                    post_samples=params['post_samples'],
                    max_spikes=params['max_spikes']
                )
                mean_wf[i,:,:] = np.nanmean(spikes[i], axis=0)
                std_wf[i,:,:] = np.nanstd(spikes[i], axis=0)
        np.save(os.path.join(params['KS_folder'], 'mean_waveforms.npy'), mean_wf)
        np.save(os.path.join(params['KS_folder'], 'std_waveforms.npy'), std_wf)
        
    print("\nDone, plotting units...")
    
    
    for i in range(n_clust):
        if cl_good[i]:
            print("\r" + str(i) + "/" + str(clusters.max()), end="")
            wf_plot = plot_wfs(i, mean_wf, std_wf, spikes[i])
            acg_plot = plot_acg(i, times_multi)
            
            name = os.path.join(params['KS_folder'], "automerge", "units", str(i) + ".pdf")
            file = PdfPages(name)
            file.savefig(wf_plot, dpi = 300)
            file.savefig(acg_plot, dpi = 300)
            file.close()

            plt.close(wf_plot)
            plt.close(acg_plot)
    
def plot_wfs(cl, mean_wf, std_wf, spikes, nchan=10, start=10, stop=60):
    peak = np.argmax(np.max(mean_wf[cl], 1) - np.min(mean_wf[cl],1))
    
    fig, a = plt.subplots(math.ceil(nchan/2), 4, figsize=(15,9))
    ch_start = max(int(peak-nchan/2), 0)
    ch_stop = min(int(peak+nchan/2), mean_wf.shape[1]-1)
    for i in range(ch_start, ch_stop):
        ind = i - (ch_start)
        mean_ind = ind+2*int(ind/2)+1
        spk_ind = mean_ind + 2
        
        # plot mean
        a = plt.subplot(math.ceil(nchan/2), 4, mean_ind)
        a.set_facecolor("black")
        plt.ylabel(i)
        plt.fill_between(range(stop-start), 
                         mean_wf[cl,i,start:stop]-2*std_wf[cl,i,start:stop], 
                         mean_wf[cl,i,start:stop]+2*std_wf[cl,i,start:stop],
                         color=(229/255,239/255,254/255),
                         alpha=0.2
                        )
        plt.plot(mean_wf[cl, i, start:stop], color=(0.3528824480447752, 0.5998034969453555, 0.9971704175788023))
        plt.ylim([(mean_wf[cl,:,start:stop]-2*std_wf[cl,:,start:stop]).min()-10, (mean_wf[cl,:,start:stop]+2*std_wf[cl,:,start:stop]).max()+10])
        
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
        
        
        # plot spikes
        a = plt.subplot(math.ceil(nchan/2), 4, spk_ind)
        a.set_facecolor("black")
        plt.ylabel(i)
        
        for j in range(min(200, spikes.shape[0])):
            plt.plot(spikes[j, i, start:stop], linewidth=.25, color=(0.3528824480447752, 0.5998034969453555, 0.9971704175788023))
            plt.ylim([(mean_wf[cl,:,start:stop]-2*std_wf[cl,:,start:stop]).min()-10, (mean_wf[cl,:,start:stop]+2*std_wf[cl,:,start:stop]).max()+10])
        
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
        
    plt.tight_layout()
    
    return fig
    
    
def plot_acg(cl, times_multi, window_size=.102, bin_size=.001):
    acg = bd.auto_correlogram(times_multi[cl]/30000, window_size, bin_size, overlap_tol=10/30000)
    
    fig, a = plt.subplots(1,1)
    plt.subplot(1,1,1)
    a.set_facecolor("black")
    a.set_yticks([0, acg.max()])
    plt.bar(range(len(acg)), acg, width=1, color=(0.3528824480447752, 0.5998034969453555, 0.9971704175788023))
    
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    
    return fig
    
if __name__ == "__main__":
    main()
        
                
                
                
                
                
        