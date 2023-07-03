import matplotlib.pyplot as plt
import numpy as np
import burst_detector as bd
import math
import os
from matplotlib.backends.backend_pdf import PdfPages

### PLOTTING CODE
def plot_merges(merges, times_multi, mean_wf, params, nchan=20, start=10, stop=60, window_size=.102, bin_size=.001):
    for merge in merges:
        merge.sort()
        wf_plot = plot_wfs(merge, mean_wf, nchan, start, stop)
        corr_plot = plot_corr(merge, times_multi, window_size, bin_size);
        
        name = os.path.join(params['KS_folder'], "automerge", "merge" + "".join(["_"+ str(cl) for cl in merge]) + ".pdf")
        file = PdfPages(name)
        file.savefig(wf_plot, dpi = 300)
        file.savefig(corr_plot, dpi = 300)
        file.close()
        
        plt.close(wf_plot)
        plt.close(corr_plot)

def plot_wfs(clust, mean_wf, nchan=20, start=10, stop=60):
    peak_chans = np.argmax(np.max(mean_wf, 2) - np.min(mean_wf,2),1)
    peak = int(np.mean(peak_chans[clust]))
    
    fig, a = plt.subplots(math.ceil(nchan/4), 4, figsize=(10,6));
    
    ch_start = int(peak-nchan/2)
    ch_stop = int(peak+nchan/2)

    lines = []
    for i in range(ch_start, ch_stop):
        a = plt.subplot(math.ceil(nchan/4), 4, i+1-(ch_start))
        a.set_facecolor("black")
        plt.ylabel(i)
        
        for cl in range(len(clust)):
            line, = plt.plot(mean_wf[clust[cl],i,start:stop], label=str(clust[cl]),color=colors[cl], alpha=0.7, linewidth=2)
            lines.append(line)

        plt.ylim([mean_wf[clust].min()-10, mean_wf[clust].max()+10])

        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        a.spines["left"].set_visible(False)

        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False
           ) # labels along the bottom edge are off
    fig.legend(lines, clust);
    
    return fig

def plot_corr(clust, times_multi, window_size=.102, bin_size=.001):
    fig, axes = plt.subplots(len(clust), len(clust), figsize=(10,5))
    
    # auto
    for i in range(len(clust)):
        acg = bd.auto_correlogram(times_multi[clust[i]]/30000, window_size, bin_size, overlap_tol=10/30000)
        
        a = plt.subplot(len(clust), len(clust), (i*len(clust))+(i+1))
        a.set_facecolor("black")
        a.set_yticks([0, acg.max()])
        plt.bar(range(len(acg)), acg, width=1, color=colors[i])
        
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    #cross
    for i in range(len(clust)):
        for j in range(len(clust)):
            if i != j:
                ccg = bd.x_correlogram(times_multi[clust[i]]/30000, 
                                       times_multi[clust[j]]/30000,
                                       window_size,
                                       bin_size)[0]
                a = plt.subplot(len(clust), len(clust), i*len(clust) + (j+1))
                a.set_facecolor("black")
                
                a.spines["top"].set_visible(False)
                a.spines["right"].set_visible(False)
                
                a.set_yticks([0, ccg.max()])
            
                if(i > j):
                    plt.bar(range(len(ccg)), ccg[::-1], width=1, color=light_colors[i])
                else:
                    plt.bar(range(len(ccg)), ccg, width=1, color=light_colors[i])
                    
                plt.xlabel
                    
                plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off

    for ax, col in zip(axes[-1], clust):
        ax.set_xlabel(col, size='large')

    for ax, row in zip(axes[:,0], clust):
        ax.set_ylabel(row, rotation=0, size='large')
      
    fig.tight_layout()
    
    return fig








## COLOR ARRAYS

light_colors = [(229/255,239/255,254/255),
                (254/255,231/255,230/255),
                (234/255,249/255,234/255),
                (254/255,250/255,229/255),
                (254/255,241/255,229/255),
                (246/255,238/255,245/255),
                (254/255,229/255,239/255),
                (249/255,242/255,235/255),
                (233/255,251/255,249/255),
                (243/255,243/255,240/255)]

colors = [(0.3528824480447752, 0.5998034969453555, 0.9971704175788023),
 (0.9832565730779054, 0.3694984452949815, 0.3488265255379734),
 (0.4666666666666667, 0.8666666666666667, 0.4666666666666667),
 (0.999, 0.8666666666666666, 0.23116059580240228),
 (0.999, 0.62745156, 0.3019607),
 (0.656421832660253, 0.35642078793464527, 0.639125729774389),
 (0.999, 0.6509803921568628, 0.788235294117647),
 (0.8352941176470589, 0.6392156862745098, 0.4470588235294118),
 (0.25098039215686274, 0.8784313725490196, 0.8156862745098039),
 (0.7098039215686275, 0.7411764705882353, 0.6470588235294118)]
