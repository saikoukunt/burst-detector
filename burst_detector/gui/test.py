import numpy as np
import pandas as pd
import os

ks_dir = r'C:\Users\Harris_Lab\Projects\burst-detector\data\rec_bank0_dense_g0\KS2.5\catgt_rec_bank0_dense_g0\rec_bank0_dense_g0_imec0\imec0_ks2'

metrics = pd.DataFrame()
seps = {".csv": ',', ".tsv": '\t'}
exclude = ["cluster_group", "waveform_metrics", "metrics"]

for file in os.listdir(ks_dir):
    if (file.endswith(".csv") or file.endswith(".tsv")) and (file[:-4] not in exclude):
        df = pd.read_csv(os.path.join(ks_dir, file), sep=seps[file[-4:]])

        if 'cluster_id' in df.columns:
            metrics = df if metrics.empty else pd.merge(metrics, df, on='cluster_id', how='outer') 

print(metrics)

# metrics['n_spikes'] = self.counts[metrics['cluster_id']]