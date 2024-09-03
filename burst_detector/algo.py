import json
import math
import os
import time
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torchvision.transforms import ToTensor

import burst_detector as bd


def run_merge(params: dict) -> tuple[str, str, str, str, str, int, int]:
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU.")

    os.makedirs(os.path.join(params["KS_folder"], "automerge"), exist_ok=True)
    os.makedirs(os.path.join(params["KS_folder"], "automerge", "merges"), exist_ok=True)

    # Load sorting and recording info.
    print("Loading files...")
    times: NDArray[np.float_] = np.load(
        os.path.join(params["KS_folder"], "spike_times.npy")
    ).flatten()
    clusters: NDArray[np.int_] = np.load(
        os.path.join(params["KS_folder"], "spike_clusters.npy")
    ).flatten()
    cl_labels: pd.DataFrame = pd.read_csv(
        os.path.join(params["KS_folder"], "cluster_group.tsv"), sep="\t"
    )
    cl_labels.set_index("cluster_id", inplace=True)

    channel_pos: NDArray[np.float_] = np.load(
        os.path.join(params["KS_folder"], "channel_positions.npy")
    )
    if "group" not in cl_labels.columns:
        cl_labels["group"] = cl_labels["KSLabel"]

    # Compute useful cluster info.
    # Load the ephys recording.
    rawData = np.memmap(params["data_filepath"], dtype=params["dtype"], mode="r")
    data: NDArray[np.int16] = np.reshape(
        rawData, (int(rawData.size / params["n_chan"]), params["n_chan"])
    )

    n_clust = clusters.max() + 1
    counts = bd.spikes_per_cluster(clusters)
    times_multi = bd.find_times_multi(
        times,
        clusters,
        np.arange(n_clust),
        data,
        params["pre_samples"],
        params["post_samples"],
    )

    # update group to noise if counts < min_spikes
    cl_labels.loc[counts < params["min_spikes"], "group"] = "noise"

    # Get cluster_ids labeled good
    good_ids = cl_labels[cl_labels["group"].isin(params["good_lbls"])].index

    cl_good = np.zeros(n_clust, dtype=bool)
    cl_good[good_ids] = True

    mean_wf, std_wf, spikes = bd.calc_mean_and_std_wf(
        params, n_clust, good_ids, times_multi, data, return_spikes=True
    )

    peak_chans = np.argmax(np.max(mean_wf, 2) - np.min(mean_wf, 2), 1)

    t0: float = time.time()

    print("Done, calculating cluster similarity...")
    sim: NDArray[np.float_] = np.ndarray(0)

    # Autoencoder-based similarity calculation.
    if params["sim_type"] == "ae":
        spk_fld: str = os.path.join(params["KS_folder"], "automerge", "spikes")
        ci: dict[str, Any] = {
            "times_multi": times_multi,
            "counts": counts,
            "good_ids": good_ids,
            "labels": cl_labels,
            "mean_wf": mean_wf,
        }
        ext_params: dict[str, Any] = {
            "spk_fld": spk_fld,
            "pre_samples": params["ae_pre"],
            "post_samples": params["ae_post"],
            "num_chan": params["ae_chan"],
            "for_shft": params["ae_shft"],
        }
        spk_snips, cl_ids = bd.generate_train_data(
            data, ci, channel_pos, ext_params, params
        )
        # Train the autoencoder if needed.
        model_path = ( 
            params["model_path"]
            if params["model_path"]
            else os.path.join(params["KS_folder"], "automerge", "ae.pt")
        )
        if not os.path.exists(model_path):
            print("Training autoencoder...")
            net, spk_data = bd.train_ae(
                spk_snips,
                cl_ids,
                do_shft=params["ae_shft"],
                num_epochs=params["ae_epochs"],
            )
            torch.save(
                net.state_dict(),
                model_path,
            )
            print(f"Autoencoder saved in {model_path}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net: bd.CN_AE = bd.CN_AE().to(device)
            net.load_state_dict(torch.load(model_path))
            net.eval()
            spk_data: bd.SpikeDataset = bd.SpikeDataset(spk_snips, cl_ids)

        # Calculate similarity using distances in the autoencoder latent space.
        sim, _, _, _ = bd.calc_ae_sim(
            mean_wf, net, peak_chans, spk_data, cl_good, do_shft=params["ae_shft"]
        )
        pass_ms = sim > params["sim_thresh"]
    elif params["sim_type"] == "mean":
        # Calculate similarity using inner products between waveforms.
        offset: NDArray[np.int_]
        wf_norms: NDArray[np.float_]
        pass_ms: NDArray[np.bool_]
        sim, offset, wf_norms, mean_wf, pass_ms = bd.calc_mean_sim(
            data, times_multi, clusters, counts, n_clust, cl_labels, mean_wf, params
        )
        sim[pass_ms == False] = 0
    pass_ms = sim > params["sim_thresh"]
    print("Found %d candidate cluster pairs" % (pass_ms.sum() / 2))
    t1: float = time.time()
    mean_sim_time: str = time.strftime("%H:%M:%S", time.gmtime(t1 - t0))

    # Calculate a significance metric for cross-correlograms.
    print("Calculating cross-correlation metric...")
    xcorr_sig: NDArray[np.float_]
    x_grams: NDArray
    shfl_xgrams: NDArray
    xcorr_sig, xgrams, shfl_xgrams = bd.calc_xcorr_metric(
        times_multi, n_clust, pass_ms, params
    )
    t4: float = time.time()
    xcorr_time: str = time.strftime("%H:%M:%S", time.gmtime(t4 - t1))
    print("Cross correlation took %s" % xcorr_time)

    # Calculate a refractor period penalty.
    print("Calculating refractory period penalty...")
    ref_pen: NDArray[np.float_]
    ref_per: NDArray[np.float_]
    ref_pen, ref_per = bd.calc_ref_p(
        times_multi, clusters, n_clust, pass_ms, xcorr_sig, params
    )
    t5: float = time.time()
    ref_pen_time: str = time.strftime("%H:%M:%S", time.gmtime(t5 - t4))
    print("Refractory period penalty took %s" % ref_pen_time)

    # Calculate the final metric.
    print("Calculating final metric...")
    final_metric: NDArray[np.float_] = np.zeros_like(sim)
    for c1 in range(n_clust):
        for c2 in range(c1, n_clust):
            met: float = (
                sim[c1, c2]
                + params["xcorr_coeff"] * xcorr_sig[c1, c2]
                - params["ref_pen_coeff"] * ref_pen[c1, c2]
            )

            final_metric[c1, c2] = max(met, 0)
            final_metric[c2, c1] = max(met, 0)

    # Calculate/perform merges.
    print("Merging...")
    channel_map: NDArray[np.int_] = np.load(
        os.path.join(params["KS_folder"], "channel_map.npy")
    ).flatten()

    old2new: dict[int, int]
    new2old: dict[int, list[int]]
    old2new, new2old = bd.merge_clusters(
        clusters, counts, mean_wf, final_metric, params
    )

    t6: float = time.time()
    merge_time: str = time.strftime("%H:%M:%S", time.gmtime(t6 - t5))

    print("Merging took %s" % merge_time)

    print("Writing to output...")
    with open(
        os.path.join(params["KS_folder"], "automerge", "old2new.json"), "w"
    ) as file:
        file.write(json.dumps(old2new, separators=(",\n", ":")))

    with open(
        os.path.join(params["KS_folder"], "automerge", "new2old.json"), "w"
    ) as file:
        file.write(json.dumps(new2old, separators=(",\n", ":")))

    merges: list[list[int]] = list(new2old.values())

    bd.plot_merges(merges, times_multi, mean_wf, std_wf, spikes, params)

    t7: float = time.time()
    total_time: str = time.strftime("%H:%M:%S", time.gmtime(t7 - t0))

    print("Total time: %s" % total_time)

    return (
        mean_sim_time,
        xcorr_time,
        ref_pen_time,
        merge_time,
        total_time,
        len(list(new2old.keys())),
        int(clusters.max()),
    )
