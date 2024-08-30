import math
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from argschema import ArgSchemaParser
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from numpy.typing import NDArray
from tqdm import tqdm

import burst_detector as bd


def main() -> None:
    from burst_detector import AutoMergeParams

    mod = ArgSchemaParser(schema_type=AutoMergeParams)
    params: dict[str, Any] = mod.args
    # TODO fix
    if params.get("max_spikes") is None:
        params["max_spikes"] = 1000

    os.makedirs(os.path.join(params["KS_folder"], "automerge"), exist_ok=True)
    os.makedirs(os.path.join(params["KS_folder"], "automerge", "units"), exist_ok=True)

    times = np.load(os.path.join(params["KS_folder"], "spike_times.npy")).flatten()
    clusters = np.load(
        os.path.join(params["KS_folder"], "spike_clusters.npy")
    ).flatten()
    n_clust = clusters.max() + 1

    # count spikes per cluster, load quality labels
    counts = bd.spikes_per_cluster(clusters)
    times_multi = bd.find_times_multi(times, clusters, np.arange(clusters.max() + 1))

    # load recording
    rawData = np.memmap(params["data_filepath"], dtype=params["dtype"], mode="r")
    data: NDArray[np.int_] = np.reshape(
        rawData, (int(rawData.size / params["n_chan"]), params["n_chan"])
    )

    # filter out low-spike/noise units
    good_ids = np.where(counts > params["min_spikes"])[0]
    mean_wf, std_wf, spikes = bd.calc_mean_and_std_wf(
        params, n_clust, good_ids, times_multi, data, return_spikes=True
    )

    for id in tqdm(good_ids, desc="Plotting units"):
        wf_plot = plot_wfs(id, mean_wf, std_wf, spikes[id])
        acg_plot = plot_acg(id, times_multi)

        name: str = os.path.join(params["KS_folder"], "automerge", f"units{id}.pdf")
        file = PdfPages(name)
        file.savefig(wf_plot, dpi=300)
        file.savefig(acg_plot, dpi=300)
        file.close()

        plt.close(wf_plot)
        plt.close(acg_plot)


def plot_wfs(
    cl: int,
    mean_wf: NDArray[np.float_],
    std_wf: NDArray[np.float_],
    spikes: NDArray[np.int_],
    nchan: int = 10,
    start: int = 10,
    stop: int = 60,
) -> Figure:
    peak: int = int(np.argmax(np.max(mean_wf[cl], 1) - np.min(mean_wf[cl], 1)))

    fig: Figure
    a: Axes | NDArray
    fig, a = plt.subplots(math.ceil(nchan / 2), 4, figsize=(15, 9))
    ch_start: int = max(int(peak - nchan / 2), 0)
    ch_stop: int = min(int(peak + nchan / 2), mean_wf.shape[1] - 1)

    for i in range(ch_start, ch_stop):
        ind: int = i - (ch_start)
        mean_ind: int = ind + 2 * int(ind / 2) + 1
        spk_ind: int = mean_ind + 2

        # plot mean
        a = plt.subplot(math.ceil(nchan / 2), 4, mean_ind)
        a.set_facecolor("black")
        plt.ylabel(str(i))
        plt.fill_between(
            range(stop - start),
            mean_wf[cl, i, start:stop] - 2 * std_wf[cl, i, start:stop],  # type: ignore
            mean_wf[cl, i, start:stop] + 2 * std_wf[cl, i, start:stop],  # type: ignore
            color=(229 / 255, 239 / 255, 254 / 255),
            alpha=0.2,
        )
        plt.plot(
            mean_wf[cl, i, start:stop],
            color=(0.3528824480447752, 0.5998034969453555, 0.9971704175788023),
        )
        plt.ylim(
            [
                (mean_wf[cl, :, start:stop] - 2 * std_wf[cl, :, start:stop]).min() - 10,
                (mean_wf[cl, :, start:stop] + 2 * std_wf[cl, :, start:stop]).max() + 10,
            ]
        )

        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        plt.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off

        # plot spikes
        a = plt.subplot(math.ceil(nchan / 2), 4, spk_ind)
        a.set_facecolor("black")
        plt.ylabel(str(i))

        for j in range(min(200, spikes.shape[0])):
            plt.plot(
                spikes[j, i, start:stop],
                linewidth=0.25,
                color=(0.3528824480447752, 0.5998034969453555, 0.9971704175788023),
            )
            plt.ylim(
                [
                    (mean_wf[cl, :, start:stop] - 2 * std_wf[cl, :, start:stop]).min()
                    - 10,
                    (mean_wf[cl, :, start:stop] + 2 * std_wf[cl, :, start:stop]).max()
                    + 10,
                ]
            )

        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        plt.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off

    plt.tight_layout()

    return fig


def plot_acg(
    cl: int, times_multi: list[NDArray[np.int_]], window_size=0.102, bin_size=0.001
) -> Figure:
    acg: NDArray[np.float_] = bd.auto_correlogram(
        times_multi[cl] / 30000, window_size, bin_size, overlap_tol=10 / 30000
    )

    fig: Figure
    a: Axes
    fig, a = plt.subplots(1, 1)
    plt.subplot(1, 1, 1)
    a.set_facecolor("black")
    a.set_yticks([0, acg.max()])

    plt.bar(
        np.array(range(len(acg))),
        acg,
        width=1,
        color=(0.3528824480447752, 0.5998034969453555, 0.9971704175788023),
    )

    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

    return fig


if __name__ == "__main__":
    main()
