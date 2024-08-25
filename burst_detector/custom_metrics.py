import logging
import math
import os
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import signal, stats
from tqdm import tqdm

import burst_detector as bd

logger = logging.getLogger("burst-detector")


# --------------------------------------------- DATA HANDLING HELPERS -------------------------------------------------
def find_times_multi(sp_times: NDArray, sp_clust: NDArray, clust_ids: list) -> list:
    """
    Finds all the spike times for each of the specified clusters.

    Args:
        sp_times (NDArray): 1-D array containing all spike times.
        sp_clust (NDArray): 1-D array containing the cluster identity of each spike.
        clust_ids (list): The cluster IDs of the desired spike times.

    Returns:
        cl_times (list): list of NumPy arrays of the spike times for each of the specified clusters.
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

    # convert inner lists to numpy arrays
    for i in range(len(cl_times)):
        cl_times[i] = np.array(cl_times[i])

    return cl_times


def extract_noise(
    data: NDArray,
    times: NDArray,
    pre_samples: int = 20,
    post_samples: int = 62,
    n_chan: int = 385,
) -> NDArray:
    """
    Extracts noise snippets from the given data based on the provided spike times.
    Args:
        data (NDArray): The input data array.
        times (NDArray): The spike times array.
        pre_samples (int, optional): The number of samples to include before each spike time. Defaults to 20.
        post_samples (int, optional): The number of samples to include after each spike time. Defaults to 62.
        n_chan (int, optional): The number of channels. Defaults to 385.
    Returns:
        NDArray: The noise snippets array.
    """
    # extraction loop
    noise = np.zeros((n_chan, 2000 * (pre_samples + post_samples)))

    ind = 0
    for i in range(1, times.shape[0]):
        # portions where no spikes occur
        if (
            (times[i] - pre_samples) - (times[i - 1] + post_samples)
        ) > pre_samples + post_samples:

            # randomly select 10 times and channels in range
            noise_times = np.random.choice(
                range(int(times[i - 1] + post_samples), int(times[i] - pre_samples)),
                10,
                replace=False,
            )

            for j in range(10):
                start = noise_times[j] - pre_samples
                end = noise_times[j] + post_samples
                snip_len = pre_samples + post_samples
                noise[:, ind * snip_len : (ind + 1) * snip_len] = data[start:end, :].T
                noise[:, ind * snip_len : (ind + 1) * snip_len] = np.nan_to_num(
                    noise[:, ind : ind + pre_samples + post_samples]
                )
                ind += 1

                if ind >= 2000:
                    return noise

    if ind < 2000 - 1:
        noise = noise[:, : (ind - 1) * (post_samples + pre_samples)]
    return noise


def calc_metrics(ks_folder: str, data_filepath: str, n_chan: int) -> None:
    """
    Calculate various metrics for spike sorting.
    Args:
        ks_folder (str): The path to the folder containing Kilosort output files.
        data_filepath (str): The path to the raw data file.
        n_chan (int): The number of channels in the raw data.
    """
    # load stuff
    times = np.load(os.path.join(ks_folder, "spike_times.npy")).flatten()
    clusters = np.load(os.path.join(ks_folder, "spike_clusters.npy")).flatten()
    n_clust = clusters.max() + 1
    times_multi = find_times_multi(times, clusters, np.arange(clusters.max() + 1))
    channel_pos = np.load(os.path.join(ks_folder, "channel_positions.npy"))

    rawData = np.memmap(data_filepath, dtype=np.int16, mode="r")
    data = np.reshape(rawData, (int(rawData.size / n_chan), n_chan))

    # skip empty ids
    good_ids = np.unique(clusters)
    cl_good = np.zeros(n_clust, dtype=bool)
    cl_good[good_ids] = True

    # TODO clean this up with your schema
    params = {
        "pre_samples": 20,
        "post_samples": 62,
        "n_chan": n_chan,
        "max_spikes": 1000,
        "KS_folder": ks_folder,
    }
    mean_wf, _, _ = bd.calc_mean_and_std_wf(
        params, n_clust, good_ids, times_multi, data, return_spikes=False
    )

    logger.info("Calculating background standard deviation...")
    noise = extract_noise(data, times, 20, 62, n_chan=n_chan)
    noise_stds = np.std(noise, axis=1)

    snrs = calc_SNR(mean_wf, noise_stds, cl_good)
    slid_rp_viols = calc_sliding_RP_viol(times_multi, cl_good, n_clust)
    num_peaks, num_troughs, wf_durs, spat_decays = calc_wf_shape_metrics(
        mean_wf, cl_good, channel_pos, 0.2
    )

    # make dataframes
    cl_ids = np.arange(n_clust)

    snr_df = pd.DataFrame({"cluster_id": cl_ids, "SNR_good": snrs})
    srv_df = pd.DataFrame({"cluster_id": cl_ids, "slid_RP_viol": slid_rp_viols})
    wf_df = pd.DataFrame(
        {
            "cluster_id": cl_ids,
            "num_peaks": num_peaks,
            "num_troughs": num_troughs,
            "wf_dur": wf_durs,
            "spat_decays": spat_decays,
        }
    )

    np.save(os.path.join(ks_folder, "mean_waveforms.npy"), mean_wf)
    # write tsv
    snr_df.to_csv(os.path.join(ks_folder, "cluster_SNR_good.tsv"), sep="\t")
    srv_df.to_csv(os.path.join(ks_folder, "cluster_RP_conf.tsv"), sep="\t")
    wf_df.to_csv(os.path.join(ks_folder, "cluster_wf_shape.tsv"), sep="\t")


def calc_SNR(mean_wf, noise_stds, cl_good):

    logger.info("Calculating peak channels and amplitudes")
    # calculate peak chans, amplitudes
    peak_chans = np.argmax(np.max(np.abs(mean_wf), axis=-1), axis=-1)
    peak_chans[peak_chans == 384] = 383
    amps = np.max(np.max(np.abs(mean_wf), axis=-1), axis=-1)

    # calculate snrs
    snrs = np.zeros(mean_wf.shape[0])
    for i in range(snrs.shape[0]):
        if cl_good[i]:
            snrs[i] = amps[i] / noise_stds[int(peak_chans[i])]

    return snrs


def max_cont(fr: float, rp: float, rec_dur: float, acc_cont: float) -> float:
    """
    Calculates the maximum number of continuous events expected based on the given parameters.
    Args:
        fr (float): The firing rate of the events.
        rp (float): The refractory period between events.
        rec_dur (float): The recording duration.
        acc_cont (float): The acceptable level of continuity.
    Returns:
        cnt_exp (float): The maximum number of continuous events expected.
    """
    time_for_viol = rp * fr * rec_dur * 2
    cnt_exp = acc_cont * time_for_viol

    return cnt_exp


def calc_sliding_RP_viol(
    times_multi: list[NDArray[np.float_]],
    cl_good: NDArray[np.bool_],
    n_clust: int,
    bin_size: float = 0.25,
    acceptThresh: float = 0.25,
) -> NDArray[np.float32]:
    """
    Calculate the sliding refractory period violation confidence for each cluster.
    Args:
        times_multi (list[NDArray[np.float_]]): A list of arrays containing spike times for each cluster.
        cl_good (NDArray[np.bool_]): An array indicating whether each cluster is considered good or not.
        n_clust (int): The number of clusters.
        bin_size (float, optional): The size of each bin in milliseconds. Defaults to 0.25.
        acceptThresh (float, optional): The threshold for accepting refractory period violations. Defaults to 0.25.
    Returns:
        NDArray[np.float32]: An array containing the refractory period violation confidence for each cluster.
    """
    b = np.arange(0, 10.25, bin_size) / 1000
    bTestIdx = np.array([1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40], dtype="int8")
    bTest = [b[i] for i in bTestIdx]  # -1 bc 0th bin corresponds to 0-0.5 ms

    RP_conf = np.zeros(n_clust, dtype=np.float32)

    for i in tqdm(range(n_clust), desc="Calculating RP viol confs"):
        times = times_multi[i] / 30000
        if cl_good[i] and times.shape[0] > 1:
            # calculate and avg halves of acg
            acg = bd.auto_correlogram(times, 2, bin_size / 1000, 5 / 30000)
            half_len = int(acg.shape[0] / 2)
            acg[half_len:] = (acg[half_len:] + acg[:half_len][::-1]) / 2
            acg = acg[half_len:]

            acg_cumsum = np.cumsum(acg)
            sum_res = acg_cumsum[bTestIdx - 1]

            # calculate max violations per refractory period size
            num_bins_2s = acg.shape[0]
            num_bins_1s = int(num_bins_2s / 2)
            bin_rate = np.mean(acg[num_bins_1s:num_bins_2s])
            max_conts = np.array(bTest) / bin_size * 1000 * bin_rate * acceptThresh

            # compute confidence of less than acceptThresh contamination at each refractory period
            confs = []
            for j, cnt in enumerate(sum_res):
                confs.append(1 - stats.poisson.cdf(cnt, max_conts[j]))
            RP_conf[i] = 1 - max(confs)

    return RP_conf


def calc_wf_shape_metrics(
    mean_wf: NDArray[np.float_],
    cl_good: NDArray[np.bool_],
    channel_pos: NDArray[np.float_],
    minThreshDetectPeaksTroughs: float = 0.2,
) -> Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.float_], NDArray[np.float_]]:
    """
    Calculate waveform shape metrics.
    Args:
        mean_wf (NDArray[np.float_]): Array of mean waveforms.
        cl_good (NDArray[np.bool_]): Array indicating whether each waveform is good or not.
        channel_pos (NDArray[np.float_]): Array of channel positions.
        minThreshDetectPeaksTroughs (float, optional): Minimum threshold to detect peaks and troughs. Defaults to 0.2.
    Returns:
        Tuple[NDArray[int], NDArray[int], NDArray[float], NDArray[float]]: A tuple containing the following metrics:
            - num_peaks: Array of the number of peaks for each waveform.
            - num_troughs: Array of the number of troughs for each waveform.
            - wf_durs: Array of waveform durations for each waveform.
            - spat_decays: Array of spatial decay values for each waveform.
    """
    peak_chans = np.argmax(np.max(np.abs(mean_wf), axis=-1), axis=-1)
    peak_chans[peak_chans >= 383] = 382

    num_peaks = np.zeros(mean_wf.shape[0], dtype="int8")
    num_troughs = np.zeros(mean_wf.shape[0], dtype="int8")
    wf_durs = np.zeros(mean_wf.shape[0], dtype=np.float32)
    spat_decays = np.zeros(mean_wf.shape[0], dtype=np.float32)

    for i in range(mean_wf.shape[0]):
        if cl_good[i]:
            peak_wf = mean_wf[i, peak_chans[i], :]

            # count peaks and troughs
            minProminence = minThreshDetectPeaksTroughs * np.max(np.abs(peak_chans))
            peak_locs, _ = signal.find_peaks(peak_wf, prominence=minProminence)
            trough_locs, _ = signal.find_peaks(-1 * peak_wf, prominence=minProminence)
            num_peaks[i] = max(peak_locs.shape[0], 1)
            num_troughs[i] = max(trough_locs.shape[0], 1)

            # calculate wf width
            peak_loc = np.argmax(peak_wf)
            trough_loc = np.argmax(-1 * peak_wf)
            wf_dur = np.abs(peak_loc - trough_loc) / 30
            wf_durs[i] = wf_dur

            # calculate spatial decay
            channels_with_same_x = np.squeeze(
                np.argwhere(
                    np.abs(channel_pos[:, 0] - channel_pos[peak_chans[i], 0]) <= 33
                )
            )
            if channels_with_same_x.shape[0] > 5:
                peak_idx = np.squeeze(
                    np.argwhere(channels_with_same_x == peak_chans[i])
                )

                if peak_idx > 5:
                    channels_for_decay_fit = channels_with_same_x[
                        peak_idx : peak_idx - 5 : -1
                    ]
                else:
                    channels_for_decay_fit = channels_with_same_x[
                        peak_idx : peak_idx + 5
                    ]

                spatialDecayPoints = np.max(
                    np.abs(mean_wf[i, channels_for_decay_fit, :]), axis=0
                )
                estimatedUnitXY = channel_pos[peak_chans[i], :]
                relativePositionsXY = (
                    channel_pos[channels_for_decay_fit, :] - estimatedUnitXY
                )
                channelDists_relative = np.sqrt(
                    np.nansum(relativePositionsXY**2, axis=1)
                )

                indSort = np.argsort(channelDists_relative)
                spatialDecayPoints_norm = spatialDecayPoints[indSort]
                spatialDecayFit = np.polyfit(
                    channelDists_relative[indSort], spatialDecayPoints_norm, 1
                )

                spat_decays[i] = spatialDecayFit[0]

    return num_peaks, num_troughs, wf_durs, spat_decays


if __name__ == "__main__":
    calc_metrics(
        r"E://T01/20240612_Tate_T01/catgt_20240612_Tate_Test_Bank0_right_g0/20240612_Tate_Test_Bank0_right_g0_imec0/imec0_ks25",
        r"E://T01/20240612_Tate_T01/catgt_20240612_Tate_Test_Bank0_right_g0/20240612_Tate_Test_Bank0_right_g0_imec0/20240612_Tate_Test_Bank0_right_g0_tcat.imec0.ap.bin",
        385,
    )
