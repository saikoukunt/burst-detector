"""
Functions and helpers to calculate normalized inner products between waveforms. 

These functions are not used in the merge finder unless the option to use mean
similarity is selected (autoencoder-based similarity is the default).
"""

import numpy as np
from numpy.typing import NDArray


def calc_wf_norms(wfs: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Calculates the Frobenius norm of each waveform in an array of waveforms.

    ### Args:
        - `wfs` (np.ndarray): Array containing waveforms. The first dimension of the
            array must index the  waveforms [i.e. shape = (# of waveforms, # channels,
            # timepoints) OR (# of waveforms, # timepoints, # channels)]

    ### Returns:
        -`wf_norm` (np.ndarray): Array of waveform norms.
    """
    wf_norms = np.zeros(wfs.shape[0])

    for i in range(wfs.shape[0]):
        wf_norms[i] = np.linalg.norm(wfs[i])

    return wf_norms


def wf_means_similarity(
    mean_wf: NDArray[np.float_],
    cl_good: NDArray[np.bool_],
    use_jitter: bool = False,
    max_jitter: int = 4,
) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.int_]]:
    """
    Calculates the normalized pairwise similarity (inner product) between pairs of
    waveforms in an array of waveforms.

    ### Args:
        - `mean_wf` (np.ndarray): Array containing unnormalized waveforms. The first dim
            of the array must index the waveforms [i.e. shape = (# of waveforms,
            # channels, # timepoints) OR (# of waveforms, # timepoints, # channels)].
        - `cl_good` (np.ndarray): Array containing cluster quality labels. Bad clusters
            will be skipped during similarity calculations.
        - `jitter` (bool): True if similarity calculations should check for time shifts
            between waveforms. Defaults to False. Note that runtime scales linearly
            with jitter_amt if enabled.
        - `jitter_amt` (int): The maximum amount of shift to search in both directions.
            Defaults to 4. Note that runtime scales linearly with jitter_amt if enabled.

    ### Returns:
        - `mean_sim` (np.ndarray): The (maximum) pairwise similarity for each pair of
            (normalized) waveforms.
        - `wf_norms` (np.ndarray): Calculated waveform norms.
        - `shifts` (np.ndarray): If jitter is enabled, the shift that produces the
            max inner product for each pair of waveforms.
    """
    n_clust: int = mean_wf.shape[0]
    mean_sim: NDArray[np.float_] = np.zeros((n_clust, n_clust))
    shifts: NDArray[np.int_] = np.zeros((n_clust, n_clust), dtype="int16")
    wf_norms: NDArray[np.float_] = calc_wf_norms(mean_wf)

    for i in range(n_clust):
        for j in range(n_clust):
            if cl_good[i] and cl_good[j]:
                if i != j:
                    norm: NDArray[np.float_] = max(wf_norms[i], wf_norms[j])
                    if norm == 0:
                        continue

                    if use_jitter:
                        sim: float
                        off: int
                        sim, off = find_jitter(mean_wf[i], mean_wf[j], max_jitter)
                        mean_sim[i, j] = sim / (norm**2)
                        shifts[i, j] = off
                    else:
                        mean_sim[i, j] = np.dot(
                            mean_wf[i].flatten(), mean_wf[j].flatten()
                        ) / (norm**2)

    return mean_sim, wf_norms, shifts


def find_jitter(
    m1: NDArray[np.float_], m2: NDArray[np.float_], max_jitter: int
) -> tuple[float, int]:
    """
    Finds the time shift that maximizes the inner product between two waveforms.

    To avoid spurious shifts, a nonzero shift must increase the inner product by < 0.1
    to be counted.

    Args:
        m1 (NDArray): Input waveform with shape (# channels, # timepoints)
        m2 (NDArray): Input waveforms with shape (# channels, # timepoints)
        max_jitter (int): The maximum shift to consider (in samples).

    Returns:
        mean_sim (float): The maximum inner product.
        jitter (int): The amount of shift that produced the maximum inner product.
            Specifies the amount that m2 shifted (negative value signifies
            m2 had to be shifted left).
    """
    jitter = 0
    mean_sim: float = np.dot(m1.flatten(), m2.flatten())
    t_length: int = m1.shape[1]

    for i in range(-1 * max_jitter, max_jitter + 1):
        if i < 0:
            c1_shift: NDArray[np.float_] = m1[:, : t_length + i]
            c2_shift: NDArray[np.float_] = m2[:, i * -1 :]
        else:
            c1_shift = m1[:, i:]
            c2_shift: NDArray[np.float_] = m2[:, : t_length - i]

        off_sim: float = np.dot(c1_shift.flatten(), c2_shift.flatten())
        if off_sim > mean_sim:
            mean_sim = off_sim
            jitter: int = i
        if (mean_sim - np.dot(m1.flatten(), m2.flatten())) < 0.1:
            mean_sim = np.dot(m1.flatten(), m2.flatten())
            jitter = 0
    return mean_sim, jitter


def cross_proj(
    c1_spikes: NDArray[np.float_],
    c2_spikes: NDArray[np.float_],
    c1_mean: NDArray[np.float_],
    c2_mean: NDArray[np.float_],
    c1_norm: float,
    c2_norm: float,
    jitter: int = 0,
) -> tuple[
    NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]
]:
    """
    Returns the cross-projections (inner products) of spikes onto normalized mean
    waveforms for a single pair of clusters.

    Args:
        c1_spikes (NDArray): Array of spike waveforms with shape (# spikes, # channels, # timepoints).
        c2_spikes (NDArray): Array of spike waveforms with shape (# spikes, # channels, # timepoints).
        c1_mean (NDArray): Cluster mean waveforms with shape (# channels, # timepoints)
        c2_mean (NDArray): Cluster mean waveforms with shape (# channels, # timepoints)
        c1_norm (float): Frobenius norms for the cluster mean waveforms.
        c2_norm (float): Frobenius norms for the cluster mean waveforms.
        jitter (float): Relative time shift between clusters, relative to c2.

    Returns:
        (proj_1on1, proj_2on1, proj_1on2, proj_2on2) (NDArray, NDArray, NDArray, NDArray): Projections
             of spikes onto their own and opposite cluster means.
    """
    t_length: int = c1_spikes.shape[2]

    if jitter < 0:
        c1_spikes = c1_spikes[:, :, : t_length + jitter]
        c1_mean = c1_mean[:, : t_length + jitter]
        c2_spikes = c2_spikes[:, :, jitter * -1 :]
        c2_mean = c2_mean[:, jitter * -1 :]
    else:
        c1_spikes = c1_spikes[:, :, jitter:]
        c1_mean = c1_mean[:, jitter:]
        c2_spikes = c2_spikes[:, :, : t_length - jitter]
        c2_mean = c2_mean[:, : t_length - jitter]

    proj_1on1: NDArray[np.float_] = np.zeros((c1_spikes.shape[0]))
    proj_2on1: NDArray[np.float_] = np.zeros((c2_spikes.shape[0]))
    proj_1on2: NDArray[np.float_] = np.zeros((c1_spikes.shape[0]))
    proj_2on2: NDArray[np.float_] = np.zeros((c2_spikes.shape[0]))

    norm: float = max(c1_norm, c2_norm)
    for i in range(c1_spikes.shape[0]):
        proj_1on1[i] = np.dot(c1_spikes[i].flatten(), c1_mean.flatten()) / (c1_norm**2)
        proj_1on2[i] = np.dot(c1_spikes[i].flatten(), c2_mean.flatten()) / (norm**2)
    for i in range(c2_spikes.shape[0]):
        proj_2on1[i] = np.dot(c2_spikes[i].flatten(), c1_mean.flatten()) / (norm**2)
        proj_2on2[i] = np.dot(c2_spikes[i].flatten(), c2_mean.flatten()) / (c2_norm**2)
    return proj_1on1, proj_2on1, proj_1on2, proj_2on2
