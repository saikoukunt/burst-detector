""" 
General utilities for ephys data-wrangling.

Assumes that ephys data is stored in the phy output format.
"""

import os
from typing import Any

import cupy as cp
import cupyx as cpx
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def find_times_multi(
    sp_times: NDArray[np.float_], sp_clust: NDArray[np.int_], clust_ids: list[int]
) -> list[NDArray[np.float_]]:
    """
    Finds all the spike times for each of the specified clusters.

    ### Args:
        - `sp_times` (np.ndarray): Spike times (in any unit of time).
        - `sp_clust` (np.ndarray): Spike cluster assignments.
        - `clust_ids` (np.ndarray): Clusters for which spike times should be returned.

    ### Returns:
        `cl_times`: found cluster spike times.
    """
    # Initialize the returned list and map cluster id to list index
    cl_times: list = []
    cl_to_ind: dict[int, int] = {}
    for i in range(len(clust_ids)):
        cl_times.append([])
        cl_to_ind[clust_ids[i]] = i

    for i in range(sp_times.shape[0]):
        if sp_clust[i] in cl_to_ind:
            cl_times[cl_to_ind[sp_clust[i]]].append(sp_times[i])
    for i in range(len(cl_times)):
        cl_times[i] = np.array(cl_times[i])

    return cl_times


def spikes_per_cluster(sp_clust: NDArray[np.int_]) -> NDArray[np.int_]:
    """
    Counts the number of spikes in each cluster.

    ### Args
        - `sp_clust` (nd.array): Spike cluster assignments.

    ### Returns
        - `counts_array` (NDArray): Number of spikes per cluster, indexed by
            cluster id. Shape is (max_cluster_id + 1,).

    """
    ids, counts = np.unique(sp_clust, return_counts=True)
    counts_array = np.zeros(ids.max() + 1, dtype=int)
    counts_array[ids] = counts
    return counts_array


def extract_spikes(
    data: NDArray[np.int_],
    times_multi: list[NDArray[np.float_]],
    clust_id: int,
    pre_samples: int = 20,
    post_samples: int = 62,
    n_chan: int = 385,
    max_spikes: int = -1,
) -> NDArray[np.int_]:
    """
    Extracts spike waveforms for the specified cluster.

    If the cluster contains more than `max_spikes` spikes, `max_spikes` random
    spikes are extracted instead.

    ### Args:
        - `data` (np.ndarray): Ephys data with shape (# of timepoints, # of channels).
            Should be passed in as an np.memmap for large datasets.
        - `times_multi` (list): Spike times indexed by cluster id.
        - `clust_id` (list): The cluster to extract spikes from
        - `pre_samples` (int): The number of samples to extract before the peak of the
            spike. Defaults to 20.
        - `post_samples` (int): The number of samples to extract after the peak of the
            spike. Defaults to 62.
        - `n_chan` (int): The number of channels in the recording. Defaults to
            385 to match NP 1.0/2.0.
        - `max_spikes` (int): The maximum number of spikes to extract. If -1, all
            spikes are extracted. Defaults to -1.

    ### Returns:
        - `spikes` (np.ndarray): Array of extracted spike waveforms with shape
            (# of spikes, # of channels, # of timepoints).
    """
    times = times_multi[clust_id].astype("int64")

    # Ignore spikes that are cut off by the ends of the recording
    while (times[0] - pre_samples) < 0:
        times = times[1:]
    while (times[-1] + post_samples) >= data.shape[0]:
        times = times[:-1]

    # Randomly pick spikes if the cluster has too many
    if (max_spikes != -1) and (max_spikes < times.shape[0]):
        np.random.shuffle(times)
        times = times[:max_spikes]

    spikes: NDArray[np.int_] = np.zeros(
        (times.shape[0], n_chan, pre_samples + post_samples), dtype="int64"
    )
    for i in range(times.shape[0]):
        spikes[i, :, :] = data[times[i] - pre_samples : times[i] + post_samples, :].T

    return spikes


def calc_mean_and_std_wf(
    params: dict[str, Any],
    n_clusters: int,
    cluster_ids: list[int],
    spike_times: list[NDArray[np.int_]],
    data: NDArray[np.int_],
    return_spikes: bool = False,
) -> NDArray:
    """
    Calculate mean waveform and std waveform for each cluster. Need to have loaded some metrics. If return_spikes is True, also returns the spike waveforms.
    Use GPU acceleration with cupy.

    Args:
        params (dict): Parameters for the recording.
        n_clusters (int): Number of clusters in the recording. Equal to the maximum cluster id + 1.
        cluster_ids (list): List of cluster ids to calculate waveforms for.
        spike_times (list): List of spike times indexed by cluster id.
        data (NDArray): Ephys data with shape (n_timepoints, n_channels).
        return_spikes (bool): Whether to return the spike waveforms. Defaults to False.

    Returns:
        NDArray: Mean waveforms for each cluster. Shape (n_clusters, n_channels, pre_samples + post_samples)
        NDArray: Std waveforms for each cluster. Shape (n_clusters, n_channels, pre_samples + post_samples)
        dict[int, NDArray]: Spike waveforms for each cluster. NDArray shape (n_spikes, n_channels, pre_samples + post_samples)
    """
    mean_wf_path = os.path.join(params["KS_folder"], "mean_waveforms.npy")
    std_wf_path = os.path.join(params["KS_folder"], "std_waveforms.npy")

    try:
        mean_wf = np.load(mean_wf_path)
        std_wf = np.load(std_wf_path)
        spikes = None
        if return_spikes:
            spikes = {}
            # Extracting spikes is faster than saving and loading them from file
            for i in tqdm(cluster_ids, desc="Loading spikes"):
                spikes_i = extract_spikes(
                    data,
                    spike_times,
                    i,
                    params["pre_samples"],
                    params["post_samples"],
                    params["n_chan"],
                    params["max_spikes"],
                )
                spikes[i] = spikes_i
    except FileNotFoundError and OSError:
        mean_wf = cpx.zeros_pinned(
            (
                n_clusters,
                params["n_chan"],
                params["pre_samples"] + params["post_samples"],
            )
        )
        std_wf = cpx.zeros_like_pinned(mean_wf)
        spikes = {}
        for i in tqdm(cluster_ids, desc="Calculating mean and std waveforms"):
            spikes_i = extract_spikes(
                data,
                spike_times,
                i,
                params["pre_samples"],
                params["post_samples"],
                params["n_chan"],
                params["max_spikes"],
            )
            spikes_cp = cp.asarray(spikes_i)
            mean_wf[i, :, :] = cp.mean(spikes_cp, axis=0)
            std_wf[i, :, :] = cp.std(spikes_cp, axis=0)
            spikes[i] = spikes_i

        print("Saving mean and std waveforms...")
        cp.save(mean_wf_path, mean_wf)
        cp.save(std_wf_path, std_wf)

        # convert to numpy
        mean_wf = cp.asnumpy(mean_wf)
        std_wf = cp.asnumpy(std_wf)
    return mean_wf, std_wf, spikes


### @internal
def get_closest_channels(
    channel_positions: NDArray[np.float_], ref_chan: int, num_close: int | None = None
) -> NDArray[np.int_]:
    """
    Gets the channels closest to a specified channel on the probe.

    ### Args:
        - `channel_positions` (np.ndarray): The XY coordinates of each channel on the
            probe (in arbitrary units)
        - `ref_chan` (int): The index of the channel to calculate distances relative to.
        - `num_close` (int, optional): The number of closest channels to return,
            including `ref_chan`.

    ### Returns:
        - `close_chans` (np.ndarray): The indices of the closest channels, sorted from
            closest to furthest. Includes the ref_chan.

    """
    x: NDArray[np.float_] = channel_positions[:, 0]
    y: NDArray[np.float_] = channel_positions[:, 1]
    x0: float
    y0: float
    x0, y0 = channel_positions[ref_chan]

    dists: NDArray[np.float_] = (x - x0) ** 2 + (y - y0) ** 2
    close_chans: NDArray[np.int_] = np.argsort(dists)
    if num_close:
        close_chans = close_chans[:num_close]
    return close_chans


def find_best_channels(
    template: NDArray[np.float_], channel_pos: NDArray[np.float_], num_close: int
) -> tuple[NDArray[np.int_], int]:
    """
    For a given waveform, finds the channels with the largest amplitude.

    ### Args:
        - `template` (np.ndarray): The waveform to find the best channels for.
        - `channel_pos` (np.ndarray): The XY coordinates of each channel on the probe
            (in arbitrary units).
        - `num_close` (int): The number of closest channels to return, including the
            peak channel (channel with largest amplitude).

    ### Returns:
        - `close_chans` (np.ndarray): The indices of the closest channels, sorted
            from closest to furthest. Includes the peak channel.
        - `peak_chan` (int): The index of the peak channel.

    """
    amplitude: NDArray[np.float_] = template.max(axis=1) - template.min(axis=1)
    peak_channel: int = min(int(np.argmax(amplitude)), 382)
    close_chans: NDArray[np.int_] = get_closest_channels(
        channel_pos, peak_channel, num_close
    )

    return close_chans, peak_channel


def get_dists(
    channel_positions: NDArray[np.float_],
    ref_chan: int,
    target_chans: NDArray[np.int_],
) -> NDArray[np.float_]:
    """
    Calculates the distance from a specified channel on the probe to a set of
    target channels.

    ### Args:
        - `channel_positions` (np.ndarray): XY coordinates of each channel on the
            probe (in arbitrary units)
        - `ref_chan` (int): The index of the channel to calculate distances relative to.
        - `target_chans` (np.ndarray): The set of target channels.

    ### Returns:
        - `dists` (np.ndarray): Distances to each of the target channels, in the same
            order as `target_chans`.
    """
    x: NDArray[np.float_] = channel_positions[:, 0]
    y: NDArray[np.float_] = channel_positions[:, 1]
    x0: float
    y0: float

    x0, y0 = channel_positions[ref_chan]
    dists: NDArray[np.float_] = (x - x0) ** 2 + (y - y0) ** 2
    dists = dists[target_chans]
    return dists
