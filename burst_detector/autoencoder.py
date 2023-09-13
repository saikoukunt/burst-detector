import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import burst_detector as bd
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

def generate_train_data(data, ci, channel_pos, gti, params, label=None, prog=None):
    """
    Generates an autoencoder training dataset from a given recording.

    Args:
        data (2d array): ephys data with shape (# of timepoints, # of channels).
            Should be passed in as an np.memmap for large datasets.
        ci (dict): contains per-cluster info --
            times (1d array):       contains all spike times.
            times_multi (list):     contains 1D arrays of spike times indexed by cluster id.
            clusters (1d array):    cluster id per cluster.
            counts (dict):          spike counts per cluster.
            labels (DataFrame):     good/mua/noise status per cluster.
            mean_wf (3d array):     mean waveforms, shape=(# of waveforms, # channels, # timepoints).
        channel_pos (2d array): contains xy coords of each channel.
        gti (dict): contains training data parameters --
            spk_fld (str):       path to folder where spike waveforms will be saved.   
            pre_samples (int):      number of samples to include before spike time. 
            post_samples (int):     number of samples to after before spike time.   
            num_chan (int):         number of channels to include in spike waveform.
            noise (bool):           True if noise snippets should be included in dataset.
            for_shft (bool):        True if dataset will be used to train a shift-invariant autoencoder.
        params (dict): JSON parameters
    Yields:
        - spike and/or noise snippets saved to spk_fld.
        - labels.csv containing file names and cluster id per snippet.
    """
    
    os.makedirs(gti['spk_fld'], exist_ok=True)
    
    # create dict of nearest channels per cluster
    chans = {}
    for i in range(ci['mean_wf'].shape[0]):
        if i in ci['counts']:
            chs, peak = bd.utils.find_best_channels(ci['mean_wf'][i], channel_pos, gti['num_chan'])
            dists =  bd.utils.get_dists(channel_pos, peak, chs)
            chans[i] = chs[np.argsort(dists)].tolist()
     
    # init extraction variables
    file_names = []
    cl_ids = []
    if gti['for_shft']:
        gti['pre_samples'] += 5
        gti['post_samples'] += 5

    outQt = (label is not None) and (prog is not None)
    if outQt:
        prog.setMaximum(ci['clusters'].max())
    
    # spike extraction loop
    tot = 0
    for i in range(ci['clusters'].max()+1):
        if (i in ci['counts']) and (ci['counts'][i] > params["min_spikes"]): # and (ci['labels'].loc[ci['labels']['cluster_id']==i, 'group'].item() == 'good'):
            cl_times = ci['times_multi'][i].astype("int64")

            # cap number of spikes
            if (params['max_spikes'] < cl_times.shape[0]):
                np.random.shuffle(cl_times)
                cl_times = cl_times[:params["max_spikes"]]

            if outQt:
                prog.setValue(i)
                label.setText("Extracting %d spikes from cluster %d/%d" % (cl_times.shape[0], i, ci['clusters'].max()))

            # save spikes to file 
            for j in range(cl_times.shape[0]):
                file_names.append("cl%d_spk%d.npy" % (i, j))
                cl_ids.append(i)
                
                start = cl_times[j]-gti['pre_samples']
                end = cl_times[j]+gti['post_samples']
                if (start > 0) and (end < data.shape[0]):
                    spike = data[start:end, chans[i]].T
                    spike = np.nan_to_num(spike)
                    out_name = gti['spk_fld'] +"/cl%d_spk%d.npy" % (i, j)
                    np.save(out_name, spike)
                    
    num_spike = len(file_names)
    
    # noise snippet extraction
    if gti['noise']:
        # dict of nearest channels per channel
        chans_2 = {}
        for i in range(channel_pos.shape[0]):
            chs = bd.utils.get_closest_channels(channel_pos, i, gti['num_chan'])
            dists = bd.utils.get_dists(channel_pos, i, chs)
            chans_2[i] = chs[np.argsort(dists)].tolist()

        # extraction loop
        ind = 0
        for i in range(1,ci['times'].shape[0]):
            # portions where no spikes occur
            if ((ci['times'][i]-gti['pre_samples']) - (ci['times'][i-1]+gti['post_samples'])) > gti['pre_samples'] + gti['post_samples']:

                # randomly select 10 times and channels in range
                noise_times = np.random.choice(range(int(ci['times'][i-1]+gti['post_samples']), int(ci['times'][i]-gti['pre_samples'])), 10, replace=False)
                noise_chs = np.random.choice(range(channel_pos.shape[0]), 10, replace=False)
                
                # save each spike to file
                for j in range(10):
                    file_names.append("cl%d_spk%d.npy" % (-1, ind))
                    cl_ids.append(-1)
                    
                    start = noise_times[j]-gti['pre_samples']
                    end = noise_times[j]+gti['post_samples']
                    noise = data[start:end, chans_2[noise_chs[j]]].T
                    noise = np.nan_to_num(noise)
                    out_name = gti['spk_fld'] + "/cl%d_spk%d.npy" % (-1, ind)
                    
                    np.save(out_name, noise)
                    ind += 1
                    
                if ind >= num_spike:
                    break
                    
    # construct and save spike labels dataframe
    df = pd.DataFrame({'file':file_names, 'cl':cl_ids}, index=None)
    df.to_csv(gti['spk_fld'] + "/labels.csv", header=False, index=False)

    if outQt:
        label.setText("Saved spikes from %d clusters." % (ci['clusters'].max()+1))

        
class SpikeDataset(Dataset):
    def __init__(self, annotations_file, spk_dir, transform=None, target_transform=None):
        self.spk_labels = pd.read_csv(annotations_file, header=None)
        self.spk_dir = spk_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.spk_labels)

    def __getitem__(self, idx):
        spk_path = os.path.join(self.spk_dir, self.spk_labels.iloc[idx, 0])
        spk = np.load(spk_path).astype('float32')
        label = self.spk_labels.iloc[idx, 1]
        if self.transform:
            spk = self.transform(spk)
        if self.target_transform:
            label = self.target_transform(label)
        return spk, label
    
    
class CN_AE(nn.Module):
    def __init__(self, imgChannels=1, n_filt=256, zDim=15, num_chan=8, num_samp=40):
        super(CN_AE, self).__init__()
        self.num_chan = num_chan
        self.num_samp = num_samp
        self.n_filt = n_filt
        self.half_filt = int(n_filt/2)
        self.qrt_filt = int(n_filt/4)
        self.featureDim = int(n_filt*self.num_chan/8*self.num_samp/8)
        
        self.encConv1 = nn.Conv2d(imgChannels, self.qrt_filt, (3,3), padding='same')
        self.encBN1 = nn.BatchNorm2d(self.qrt_filt)
        self.encPool1 = nn.MaxPool2d((2,2), return_indices=True)
        
        self.encConv2 = nn.Conv2d(self.qrt_filt, self.half_filt, (3,3), padding='same')
        self.encBN2 = nn.BatchNorm2d(self.half_filt)
        self.encPool2 = nn.MaxPool2d((2,2), return_indices=True)
        
        self.encConv3 = nn.Conv2d(self.half_filt, n_filt, (3,3), padding='same')
        self.encBN3 = nn.BatchNorm2d(n_filt)
        self.encPool3 = nn.MaxPool2d((2,2), return_indices=True)
        
        self.encFC1 = nn.Linear(self.featureDim, zDim)
        self.decFC1 = nn.Linear(zDim, self.featureDim)
        self.decBN1 = nn.BatchNorm2d(n_filt)
        
        self.decUpSamp1 = nn.Upsample((int(num_chan/4), int(num_samp/4)))
        self.decConv1 = nn.ConvTranspose2d(n_filt, self.half_filt, (3,3), padding=(1,1))
        self.decBN2 = nn.BatchNorm2d(self.half_filt)
        
        self.decUpSamp2 = nn.Upsample((int(num_chan/2), int(num_samp/2)))
        self.decConv2 = nn.ConvTranspose2d(self.half_filt, self.qrt_filt, (3,3), padding=(1,1))
        self.decBN3 = nn.BatchNorm2d(self.qrt_filt)
        
        self.decUpSamp3 = nn.Upsample((num_chan, num_samp))
        self.decConv3 = nn.ConvTranspose2d(self.qrt_filt, imgChannels, (3,3), padding=(1,1))

    def encoder(self, x):
        x = F.relu(self.encBN1(self.encConv1(x)))
        x, self.encInd1 = self.encPool1(x)
        
        x = F.relu(self.encBN2(self.encConv2(x)))
        x, self.encInd2 = self.encPool2(x)
        
        x = F.relu(self.encBN3(self.encConv3(x)))
        x, self.encInd3 = self.encPool3(x)
        
        x = x.view(-1, self.featureDim)
        
        x = self.encFC1(x)
        
        return x

    def decoder(self, z):
        x = self.decFC1(z)
        x = x.view(-1, self.n_filt, int(self.num_chan/8), int(self.num_samp/8))
        x = F.relu(self.decBN1(x))
        
        x = self.decUpSamp1(x)
        x = F.relu(self.decBN2(self.decConv1(x)))
        
        x = self.decUpSamp2(x)
        x = F.relu(self.decBN3(self.decConv2(x)))
        
        x = self.decUpSamp3(x)
        x = self.decConv3(x)
        
        return x

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        
        return out
    
        
def train_ae(spk_fld, counts, n_filt=256, num_epochs=10, zDim=15, lr=1e-3, pre_samples=10, post_samples=30, model=None, 
             do_shft=False, label=None, prog=None):
    """
    Trains an autoencoder on the given spike dataset.

    Args:
        spk_fld (str): path to folder containing dataset.
        counts (dict): spike counts per cluster.
    KWArgs:
        num_epochs (int): number of training epochs.
        zDim (int): latent dimensionality of CN_AE.
        lr (float): optimizer learning rate.
        pre_samples (int): number of samples included before spike time. 
        post_samples (int): number of samples included after spike time. 
        model (nn.Module): custom model, if using.
        do_shft (bool): True if training samples should be randomly time-shifted.
    Returns:
        net (nn.Module): trained network.
        spk_data (SpikeDataset): dataset that was used for training.
    """
                                    
    # load dataset 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                               
    spk_data = SpikeDataset(spk_fld +"/labels.csv", spk_fld +"/", ToTensor())
    labels = spk_data.spk_labels.iloc[:, 1]

    # train-test split
    train_indices, test_indices, _, _ = train_test_split(
        range(len(spk_data)),
        labels,
        stratify=labels,
        test_size=0.2,
        random_state=42
    )
    train_split = Subset(spk_data, train_indices)
    test_split = Subset(spk_data, test_indices)
    
    try:
        wt_noise = 1/len(labels.loc[labels == -1])
    except ZeroDivisionError:
        wt_noise = None

    # sample weighting
    # sample_weights = [wt_noise if label == -1 else 1/counts[int(label)] for label in labels[train_indices]]
    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_split), replacement=True)
    BATCH_SIZE = 128
    # train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, sampler=sampler)
    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_split, batch_size=BATCH_SIZE)
    
    # init network, params                                 
    net = model if model else CN_AE(zDim=zDim, n_filt=n_filt).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    test_loss = np.zeros(num_epochs)

    outQt = (label is not None) and (prog is not None)
    if outQt:
        prog.setMaximum(num_epochs*(len(train_loader) + len(test_loader)))
    
    # training loop
    for epoch in range(num_epochs):     
        print('Epoch %d/%d' % (epoch+1, num_epochs))
        running_loss = 0
        last_loss = 0
        net.train()

        for idx, data in enumerate(train_loader, 0):
            print("\rTraining batch " + str(idx+1) +'/' + str(len(train_loader)), end="")
            if outQt:
                label.setText('EPOCH %d/%d - train batch %d/%d' % (epoch+1, num_epochs, idx+1, len(train_loader)))
                prog.setValue(epoch*(len(train_loader) + len(test_loader)) + idx)
             
            # pre-train modifications                     
            spks, cl = data
            if do_shft:
                targ = spks[:,:,:,5:-5].clone().to(device)

                # randomly time-shift 30% of spikes
                shft_ind = np.random.choice(np.arange(spks.shape[0]), int(spks.shape[0]*.3), replace=False)
                shft_spks = spks[:,:,:,5:-5].clone().to(device)
                for ind in shft_ind:
                    shift = np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5], 1)[0]
                    shft_spks[ind,:,:,:] = spks[ind,:,:,5+shift:pre_samples+post_samples+5+shift].clone().to(device)
                spks = shft_spks
            else:
                targ = spks.to(device)
                spks = spks.to(device)
            targ[cl==-1,:,:,:] = 0  # noise snippets should be reconstructed as 0

            # backprop
            out = net(spks)
            loss = loss_fn(out, targ)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # report train loss
            running_loss += loss
            if idx % 500 == 499:
                last_loss = running_loss/500
                print(" Batch %d | loss: %.4f" % (idx, last_loss))
                running_loss = 0
        print()
        last_loss = running_loss/(len(train_loader) % 500)
                                    
        # calculate test loss
        running_tloss = 0
        net.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                print("\rTesting batch " + str(i+1) + "/" + str(len(test_loader)), end="")
                if outQt:
                    label.setText('EPOCH %d/%d - test batch %d/%d' % (epoch+1, num_epochs, i+1, len(test_loader)))
                    prog.setValue(epoch*(len(train_loader) + len(test_loader)) + len(train_loader) + i)

                # pre-test modifications
                spks, cl = data
                if do_shft:
                    targ = spks[:,:,:,5:-5].clone().to(device)

                    # randomly time-shift 30% of spikes
                    shft_ind = np.random.choice(np.arange(spks.shape[0]), int(spks.shape[0]*.3), replace=False)
                    shft_spks = spks[:,:,:,5:-5].clone().to(device)
                    for ind in shft_ind:
                        shift = np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5], 1)[0]
                        shft_spks[ind,:,:,:] = spks[ind,:,:,5+shift:pre_samples+post_samples+5+shift].clone().to(device)
                    spks = shft_spks
                else:
                    targ = spks.to(device)
                    spks = spks.to(device)
                targ[cl==-1,:,:,:] = 0  # noise snippets should be reconstructed as 0

                out = net(spks)
                tloss = loss_fn(out, targ)
                running_tloss += tloss

        avg_vloss = running_tloss/(i+1)
        test_loss[epoch] = avg_vloss

        print("\nLOSS | train: %.4f | test %.4f" % (last_loss, avg_vloss))

    if outQt:
        prog.setValue(num_epochs*(len(train_loader) + len(test_loader)))
        label.setText("Done training.")
        
    return net, spk_data