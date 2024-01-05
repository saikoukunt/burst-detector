from burst_detector.utils import find_times_multi, extract_spikes, spikes_per_cluster, find_best_channels, get_dists
from burst_detector.cluster_metrics import wf_means_similarity, calc_wf_norms, cross_proj
from burst_detector.xcorr import bin_spike_trains, x_correlogram, auto_correlogram, xcorr_sig
from burst_detector.stages import calc_mean_sim, calc_ae_sim, calc_cross_sim, calc_xcorr_metric, calc_ref_p, merge_clusters
from burst_detector.algo import run_merge
from burst_detector.bursts import find_bursts, base_algo
from burst_detector.schemas import AutoMergeParams, OutputParams
from burst_detector.eval_metrics import calc_fr_unif
from burst_detector.autoencoder import generate_train_data, train_ae, CN_AE, SpikeDataset
from burst_detector.plot import plot_merges