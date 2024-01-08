import math
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from argschema import ArgSchemaParser
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from numpy.typing import NDArray

import burst_detector as bd


def main() -> None:
    from burst_detector import AutoMergeParams

    mod = ArgSchemaParser(schema_type=AutoMergeParams)
    params: dict[str, Any] = mod.args

    os.makedirs(os.path.join(params["KS_folder"], "automerge"), exist_ok=True)
    os.makedirs(os.path.join(params["KS_folder"], "automerge", "units"), exist_ok=True)

    times: NDArray[np.float_] = np.load(
        os.path.join(params["KS_folder"], "spike_times.npy")
    ).flatten()
    clusters: NDArray[np.int_] = np.load(
        os.path.join(params["KS_folder"], "spike_clusters.npy")
    ).flatten()
    n_clust: int = clusters.max() + 1

    # count spikes per cluster, load quality labels
    counts: dict[int, int] = bd.spikes_per_cluster(clusters)
    times_multi: list[NDArray[np.float_]] = bd.find_times_multi(
        times, clusters, np.arange(clusters.max() + 1)
    )

    # load recording
    rawData = np.memmap(params["data_filepath"], dtype=params["dtype"], mode="r")
    data: NDArray[np.int_] = np.reshape(
        rawData, (int(rawData.size / params["n_chan"]), params["n_chan"])
    )

    # filter out low-spike/noise units
    cl_good: NDArray[np.bool_] = np.zeros(n_clust, dtype=bool)
    unique: NDArray[np.int_] = np.unique(clusters)
    for i in range(n_clust):
        if (i in unique) and (counts[i] > params["min_spikes"]):
            cl_good[i] = True
    try:
        mean_wf: NDArray[np.float_] = np.load(
            os.path.join(params["KS_folder"], "mean_waveforms.npy")
        )
        std_wf: NDArray[np.float_] = np.load(
            os.path.join(params["KS_folder"], "std_waveforms.npy")
        )

        print("Caching spikes...")
        spikes: dict[int, NDArray[np.int_]] = {}
        for i in range(n_clust):
            print("\r" + str(i) + "/" + str(clusters.max()), end="")
            if cl_good[i]:
                spikes[i] = bd.extract_spikes(
                    data,
                    times_multi,
                    i,
                    n_chan=params["n_chan"],
                    pre_samples=params["pre_samples"],
                    post_samples=params["post_samples"],
                    max_spikes=params["max_spikes"],
                )
    except OSError:
        print(
            "mean_waveforms.npy doesn't exist, calculating mean waveforms on the fly..."
        )
        mean_wf = np.zeros(
            (n_clust, params["n_chan"], params["pre_samples"] + params["post_samples"])
        )
        std_wf = np.zeros_like(mean_wf)
        spikes = {}

        for i in range(n_clust):
            print("\r" + str(i) + "/" + str(clusters.max()), end="")
            if cl_good[i]:
                spikes[i] = bd.extract_spikes(
                    data,
                    times_multi,
                    i,
                    n_chan=params["n_chan"],
                    pre_samples=params["pre_samples"],
                    post_samples=params["post_samples"],
                    max_spikes=params["max_spikes"],
                )
                mean_wf[i, :, :] = np.nanmean(spikes[i], axis=0)
                std_wf[i, :, :] = np.nanstd(spikes[i], axis=0)
        np.save(os.path.join(params["KS_folder"], "mean_waveforms.npy"), mean_wf)
        np.save(os.path.join(params["KS_folder"], "std_waveforms.npy"), std_wf)

    print("\nDone, plotting units...")

    for i in range(n_clust):
        if cl_good[i]:
            print("\r" + str(i) + "/" + str(clusters.max()), end="")
            wf_plot: Figure = plot_wfs(i, mean_wf, std_wf, spikes[i])
            acg_plot: Figure = plot_acg(i, times_multi)

            name: str = os.path.join(
                params["KS_folder"], "automerge", "units", str(i) + ".pdf"
            )
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
