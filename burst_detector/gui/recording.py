import json
import logging
import os

import numpy as np
import pandas as pd
from argschema import ArgSchemaParser

import burst_detector as bd

logger = logging.getLogger("burst-detector")


class Recording(object):

    def __init__(self, filename):
        self.ks_dir = os.path.dirname(filename)

        # parse JSON?
        try:
            json_file = open(os.path.join(self.ks_dir, "input.json"))
            self.params = json.load(json_file)
        except FileNotFoundError:
            logger.info("No JSON found, using default SpECtr parameters!")
            self.params = {}

        self.params = ArgSchemaParser(
            input_data=self.params, schema_type=bd.schemas.AutomergeGUIParams
        ).args

        logger.info(self.params)

        with open(filename, "r") as param_file:
            for line in param_file:
                elem = line.split(sep="=")
                self.params[elem[0].strip()] = eval(elem[1].strip())

        self.spikes = {}
        self._load_recording()

    def _load_recording(self):
        # load required files (times, clusters, channel_pos, labels, spike_templates, similar_templates)
        self.sp_times = np.load(os.path.join(self.ks_dir, "spike_times.npy")).flatten()
        try:
            self.clusters = np.load(
                os.path.join(self.ks_dir, "spike_clusters.npy")
            ).flatten()
        except FileNotFoundError:
            self.clusters = np.load(
                os.path.join(self.ks_dir, "spike_templates.npy")
            ).flatten()
        self.cl_labels = pd.read_csv(
            os.path.join(self.ks_dir, "cluster_group.tsv"), sep="\t"
        )
        self.channel_pos = np.load(os.path.join(self.ks_dir, "channel_positions.npy"))
        self.spike_templates = np.load(
            os.path.join(self.ks_dir, "spike_templates.npy")
        ).flatten()
        self.similar_templates = np.load(
            os.path.join(self.ks_dir, "similar_templates.npy")
        )
        self.n_templates = self.similar_templates.shape[0]

        # load data
        self.rawData = np.memmap(
            os.path.join(self.ks_dir, self.params["dat_path"]),
            dtype=self.params["dtype"],
            mode="r",
        )
        self.data = np.reshape(
            self.rawData,
            (
                int(self.rawData.size / self.params["n_channels_dat"]),
                self.params["n_channels_dat"],
            ),
        )

        # calculate things
        self.n_clust = self.clusters.max() + 1
        self.counts = bd.spikes_per_cluster(self.clusters, self.params["max_spikes"])
        self.cl_times = bd.find_times_multi(
            self.sp_times,
            self.clusters,
            np.arange(self.n_clust),
            self.params["max_spikes"],
            self.data,
        )
        self.cl_inds = np.unique(self.clusters)
        self.cl_templates = {}
        self._calc_temp_counts()

        # try loading mean_wf
        self.mean_wf, _, _ = bd.calc_mean_and_std_wf(
            self.params, self.n_clust, self.cl_inds, self.cl_times, self.data
        )
        self.peak_chans = np.argmax(
            np.max(self.mean_wf, 2) - np.min(self.mean_wf, 2), 1
        )

        # arrays for similarities
        self.cl_temp_sim = -1 * np.ones((self.n_clust, self.n_clust))
        self.cl_ae_sim = -1 * np.ones((self.n_clust, self.n_clust))

        # load cluster metrics
        self.cluster_metrics = self._load_cluster_metrics()

        logger.info(f"Loaded KS output from {self.ks_dir}")

    def _load_cluster_metrics(self):
        metrics = pd.DataFrame()

        seps = {".csv": ",", ".tsv": "\t"}
        exclude = ["waveform_metrics", "metrics"]

        for file in os.listdir(self.ks_dir):
            if (file.endswith(".csv") or file.endswith(".tsv")) and (
                file[:-4] not in exclude
            ):
                df = pd.read_csv(os.path.join(self.ks_dir, file), sep=seps[file[-4:]])

                if "cluster_id" in df.columns:
                    new_feat = [col for col in df.columns if col not in metrics.columns]
                    metrics = (
                        df
                        if metrics.empty
                        else pd.merge(
                            metrics,
                            df[["cluster_id"] + new_feat],
                            on="cluster_id",
                            how="outer",
                        )
                    )
        metrics.set_index("cluster_id", inplace=True)
        metrics["n_spikes"] = self.counts[metrics.index]

        metrics["cur_label"] = ""

        cols = ["cluster_id", "ch", "sh", "n_spikes", "cur_label", "KSLabel"]
        metrics = metrics[cols + [c for c in metrics.columns if c not in cols]]
        metrics["ch"] = self.peak_chans[metrics["cluster_id"].astype("int")]

        # Copy KSLabel to group if group.tsv doesn't exist
        if "group" not in metrics.columns:
            metrics["group"] = metrics["KSLabel"]

        return metrics

    def _calc_temp_counts(self):
        for i in range(self.n_clust):
            if self.counts[i] > 0:
                spike_ids = self.cl_inds[i]
                temps = self.spike_templates[spike_ids]
                temp_counts = np.bincount(temps, minlength=self.n_templates)
                temp_cl = np.nonzero(temp_counts)[0]
                self.cl_templates[i] = temp_cl

    # def _sing_temp_counts(self, id):
    #     spike_ids = self.cl_inds[id]
    #     temps = self.spike_templates[spike_ids]
    #     temp_counts = np.bincount(temps, minlength=self.n_templates)
    #     temp_cl = np.nonzero(temp_counts)[0]
    #     return temp_cl

    def template_similarity(self, clust_id):
        # return pre-calculated values if they exist
        if self.cl_temp_sim[clust_id, 0] != -1:
            return self.cl_temp_sim[clust_id, :]

        sims = np.max(
            self.similar_templates[self.cl_templates[clust_id], :], axis=0
        )  # max similarity of cluster to each template

        def _sim_ij(cj):
            if self.counts[cj] == 0:
                return 0
            if cj < self.n_templates:
                return sims[cj]
            return np.max(
                sims[self.cl_templates[cj]]
            )  # max similarity between all pairs of templates

        for j in range(self.n_clust):
            self.cl_temp_sim[clust_id, j] = _sim_ij(j)

        self.cl_temp_sim[clust_id, :] = np.nan_to_num(self.cl_temp_sim[clust_id, :])

        self.cl_temp_sim[clust_id, self.cl_temp_sim[clust_id, :] < 0.001] = 0

        return self.cl_temp_sim[clust_id, :]

    def merge(self, cl_list):
        # new cluster id
        self.n_clust += 1
        new_id = self.n_clust - 1

        # update sp_clusters
        for i in range(len(cl_list)):
            for ind in self.cl_inds[i]:
                self.clusters[ind] = new_id

        # update cl_times
        temp = self.cl_times[cl_list[0]]
        for i in range(1, len(cl_list)):
            temp = np.concatenate((temp, self.cl_times[cl_list[i]]), axis=0)
        self.cl_times.append(temp)  # test if this works in a notebook

        # update cl_inds
        temp = self.cl_inds[cl_list[0]]
        for i in range(1, len(cl_list)):
            temp = np.concatenate((temp, self.cl_inds[cl_list[i]]), axis=0)
        self.cl_inds.append(temp)  # test if this works in a notebook

        # cl_templates
        temp = list(self.cl_templates[cl_list[0]])
        for i in range(1, len(cl_list)):
            temp += list(self.cl_templates[cl_list[i]])

        self.cl_templates[new_id] = temp
        logger.info(self.cl_templates[new_id])

        # spikes
        temp = self.spikes[cl_list[0]]
        for i in range(1, len(cl_list)):
            temp = np.concatenate((temp, self.spikes[cl_list[i]]), axis=0)
        self.spikes[new_id] = temp

        # mean_wf
        temp = self.counts[cl_list] / sum(self.counts[cl_list]) * self.mean_wf[cl_list]
        self.mean_wf = np.concatenate((self.mean_wf, temp), axis=0)

        # expand sim arrays
        temp = -1 * np.ones((1, self.n_clust - 1))
        self.cl_temp_sim = np.concatenate((self.cl_temp_sim, temp), axis=0)
        self.cl_ae_sim = np.concatenate((self.cl_ae_sim, temp), axis=0)

        temp = -1 * np.ones((self.n_clust, 1))
        self.cl_temp_sim = np.concatenate((self.cl_temp_sim, temp), axis=1)
        self.cl_ae_sim = np.concatenate((self.cl_ae_sim, temp), axis=1)

        return new_id

    def load_spikes(self, cl):
        if cl in self.spikes:
            return
        self.spikes[cl] = bd.extract_spikes(
            self.data,
            self.cl_times,
            self.clusters,
            cl,
            n_chan=self.params["n_channels_dat"],
            max_spikes=500,
        )

        # if self.calc_means:
        #     self.mean_wf[cl] = np.nanmean(self.spikes[cl], axis=0)
        #     self.std_wf[cl] = np.nanstd(self.spikes[cl], axis=0)
