import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from argschema import ArgSchemaParser
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import burst_detector as bd


def main() -> None:
    from burst_detector import AutoMergeParams

    mod = ArgSchemaParser(schema_type=AutoMergeParams)
    params: dict[str, Any] = mod.args
    # TODO fix
    if params.get("max_spikes") is None:
        params["max_spikes"] = 1000

    automerge = os.path.join(params["KS_folder"], "automerge")
    os.makedirs(os.path.join(automerge, "units"), exist_ok=True)

    times = np.load(os.path.join(params["KS_folder"], "spike_times.npy")).flatten()
    clusters = np.load(
        os.path.join(params["KS_folder"], "spike_clusters.npy")
    ).flatten()
    n_clust = clusters.max() + 1

    # load recording
    rawData = np.memmap(params["data_filepath"], dtype=params["dtype"], mode="r")
    data = np.reshape(rawData, (int(rawData.size / params["n_chan"]), params["n_chan"]))

    # count spikes per cluster, load quality labels
    counts = bd.spikes_per_cluster(clusters)
    times_multi = bd.find_times_multi(times, clusters, np.arange(n_clust), data)

    # filter out low-spike/noise units
    good_ids = np.where(counts > params["min_spikes"])[0]
    mean_wf, std_wf, spikes = bd.calc_mean_and_std_wf(
        params, n_clust, good_ids, times_multi, data, return_spikes=True
    )

    for id in tqdm(good_ids, desc="Plotting units"):
        wf_plot = bd.plot_wfs([id], mean_wf, std_wf, spikes)
        acg_plot = bd.plot_corr([id], times_multi, params)

        name = os.path.join(params["KS_folder"], "automerge", f"units{id}.pdf")
        with PdfPages(name) as file:
            file.savefig(wf_plot, dpi=300)
            file.savefig(acg_plot, dpi=300)

        plt.close(wf_plot)
        plt.close(acg_plot)


if __name__ == "__main__":
    main()
