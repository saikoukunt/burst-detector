from typing import Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import random
import os
import burst_detector as bd
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

def generate_train_data(
        data: NDArray[np.int_], 
        ci: dict[str, Any], 
        channel_pos: NDArray[np.float_], 
        gti: dict[str, Any],
        params: dict[str, Any]
    ) -> tuple[torch.Tensor, NDArray[np.int_]]:
    """
    Generates an autoencoder training dataset from a given recording.

    Args:
        data (np.ndarray): ephys data with shape (# of timepoints, # of channels).
            Should be passed in as an np.memmap for large datasets.
        ci (dict): contains per-cluster info --
            times (np.ndarray):       contains all spike times.
            times_multi (list):     contains 1D arrays of spike times indexed by cluster id.
            clusters (np.ndarray):    cluster id per cluster.
            counts (dict):          spike counts per cluster.
            labels (DataFrame):     good/mua/noise status per cluster.
            mean_wf (np.ndarray):     mean waveforms, shape=(# of waveforms, # channels, # timepoints).
        channel_pos (np.ndarray): contains xy coords of each channel.
        gti (dict): contains training data parameters --
            pre_samples (int):      number of samples to include before spike time. 
            post_samples (int):     number of samples to after before spike time.   
            num_chan (int):         number of channels to include in spike waveform.
            for_shft (bool):        True if dataset will be used to train a shift-invariant autoencoder.
        params (dict): JSON parameters
    Returns:
        spikes (torch.Tensor): extracted spike snippets with shape (# snippets, 1, # channels, # timepoints)
        cl_ids (np.ndarray): cluster labels of spike snippets with shape (# snippets)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # create dict of nearest channels per cluster
    chans: dict[int, list[int]] = {}
    for i in range(ci['mean_wf'].shape[0]):
        if i in ci['counts']:
            chs: NDArray[np.int_]; peak: int
            chs, peak = bd.find_best_channels(ci['mean_wf'][i], channel_pos, gti['num_chan'])
            dists: NDArray[np.float_] =  bd.get_dists(channel_pos, peak, chs)
            chans[i] = chs[np.argsort(dists)].tolist()
     
    # init extraction variables
    if gti['for_shft']:
        gti['pre_samples'] += 5
        gti['post_samples'] += 5
        
    # pre-count number of snippets
    n_snip: int = 0
    for i in range(ci['clusters'].max()+1):
        if (i in ci['counts']) and (ci['counts'][i] > params["min_spikes"]) and (ci['labels'].loc[ci['labels']['cluster_id']==i, 'group'].item() in params['good_lbls']):
            cl_times: NDArray[np.int64] = ci['times_multi'][i].astype("int64")

            # cap number of spikes
            n_snip += min(params['max_spikes'], cl_times.shape[0])
            
    # extract snippets
    snip_ind: int = 0
    spikes: torch.Tensor = torch.zeros((n_snip, 1, 8, gti['pre_samples']+gti['post_samples'])).to(device)
    cl_ids: NDArray[np.int16] = np.zeros(n_snip, dtype='int16')
    for i in range(ci['clusters'].max()+1):
        if (i in ci['counts']) and (ci['counts'][i] > params["min_spikes"]) and (ci['labels'].loc[ci['labels']['cluster_id']==i, 'group'].item() in params['good_lbls']):
            cl_times = ci['times_multi'][i].astype("int64")

            # cap number of spikes
            if (params['max_spikes'] < cl_times.shape[0]):
                np.random.shuffle(cl_times)
                cl_times = cl_times[:params['max_spikes']]

            # save snippets to array
            for j in range(cl_times.shape[0]):
                cl_ids[snip_ind+j] = i

                start: int = cl_times[j]-gti['pre_samples']
                end: int = cl_times[j]+gti['post_samples']
                spike: NDArray[np.float_] = np.nan_to_num(data[start:end, chans[i]].T)
                spikes[snip_ind+j] = torch.Tensor(spike).unsqueeze(dim=0) 

            snip_ind += cl_times.shape[0]
    
    return spikes, cl_ids

class SpikeDataset(Dataset):
    def __init__(self, 
            spikes: torch.Tensor, 
            labels: NDArray[np.int_], 
            transform: Callable | None = None, 
            target_transform: Callable | None = None
        ) -> None:
        
        self.spikes: torch.Tensor = spikes
        self.labels: NDArray[np.int_] = labels
        self.transform: Callable | None = transform
        self.target_transform: Callable | None = target_transform

    def __len__(self) -> int:
        return self.spikes.shape[0]

    def __getitem__(self, idx: int)-> tuple[torch.Tensor, int]:
        spk: torch.Tensor = self.spikes[idx]
        label: int = self.labels[idx]
        if self.transform:
            spk = self.transform(spk)
        if self.target_transform:
            label = self.target_transform(label)
        return spk, label
    
    
class CN_AE(nn.Module):
    def __init__(self, 
        imgChannels: int = 1, 
        n_filt: int = 256, 
        zDim: int = 15, 
        num_chan: int = 8, 
        num_samp: int = 40
    ) -> None:
        super(CN_AE, self).__init__()
        self.num_chan: int = num_chan
        self.num_samp: int = num_samp
        self.n_filt: int = n_filt
        self.half_filt = int(n_filt/2)
        self.qrt_filt = int(n_filt/4)
        self.featureDim = int(n_filt*self.num_chan/8*self.num_samp/8)
        
        self.encConv1 = nn.Conv2d(imgChannels, self.qrt_filt, (3,3), padding='same')
        self.encBN1 = nn.BatchNorm2d(self.qrt_filt)
        self.encPool1 = nn.MaxPool2d((2,2))
        
        self.encConv2 = nn.Conv2d(self.qrt_filt, self.half_filt, (3,3), padding='same')
        self.encBN2 = nn.BatchNorm2d(self.half_filt)
        self.encPool2 = nn.MaxPool2d((2,2))
        
        self.encConv3 = nn.Conv2d(self.half_filt, n_filt, (3,3), padding='same')
        self.encBN3 = nn.BatchNorm2d(n_filt)
        self.encPool3 = nn.MaxPool2d((2,2))
        
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

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.encBN1(self.encConv1(x)))
        x = self.encPool1(x)
        
        x = F.relu(self.encBN2(self.encConv2(x)))
        x = self.encPool2(x)
        
        x = F.relu(self.encBN3(self.encConv3(x)))
        x = self.encPool3(x)
        
        x = x.view(-1, self.featureDim)
        x = self.encFC1(x)
        
        return x

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.decFC1(z)
        x = x.view(-1, self.n_filt, int(self.num_chan/8), int(self.num_samp/8))
        x = F.relu(self.decBN1(x))
        
        x = self.decUpSamp1(x)
        x = F.relu(self.decBN2(self.decConv1(x)))
        
        x = self.decUpSamp2(x)
        x = F.relu(self.decBN3(self.decConv2(x)))
        
        x = self.decUpSamp3(x)
        x = self.decConv3(x)
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self.encoder(x)
        out: torch.Tensor = self.decoder(z)
        
        return out
    
        
def train_ae(
        spikes: torch.Tensor, 
        cl_ids: NDArray[np.int_],
        n_filt: int = 256, 
        num_epochs: int = 25, 
        zDim: int = 15, 
        lr: float = 1e-3, 
        pre_samples: int = 10, 
        post_samples: int = 30, 
        model: CN_AE|None = None, 
        do_shft: bool =False
    ) -> tuple[CN_AE, SpikeDataset]:
    """
    Trains an autoencoder on the given spike dataset.

    Args:
        spikes (torch.Tensor): spike snippets.
        cl_ids (np.ndarray): cluster ids of spike snippets.
    KWArgs:
        num_epochs (int): number of training epochs.
        zDim (int): latent dimensionality of CN_AE.
        lr (float): optimizer learning rate.
        pre_samples (int): number of samples included before spike time. 
        post_samples (int): number of samples included after spike time. 
        model (nn.Module): custom model, if using.
        do_shft (bool): True if training samples should be randomly time-shifted.
    Returns:
        net (CN_AE): trained network.
        spk_data (SpikeDataset): dataset that was used for training.
    """
                                    
    # load dataset 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      
    spk_data = SpikeDataset(spikes, cl_ids)     
    labels = cl_ids                    

    # train-test split
    train_indices: list; test_indices: list; _: Any
    train_indices, test_indices, _, _ = train_test_split(
        range(len(spk_data)),
        labels,
        stratify=labels,
        test_size=0.2,
        random_state=42
    )
    train_split = Subset(spk_data, train_indices)
    test_split = Subset(spk_data, test_indices)

    BATCH_SIZE: int = 128
    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_split, batch_size=BATCH_SIZE)
    
    # init network, params                                 
    net: CN_AE = model if model else CN_AE(zDim=zDim, n_filt=n_filt).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    test_loss: NDArray[np.float_] = np.zeros(num_epochs)
    
    # training loop
    for epoch in range(num_epochs):     
        print('Epoch %d/%d' % (epoch+1, num_epochs))
        running_loss: float | torch.Tensor = 0
        last_loss: float | torch.Tensor = 0
        net.train()

        for idx, data in enumerate(train_loader, 0):
            print("\rTraining batch " + str(idx+1) +'/' + str(len(train_loader)), end="")
             
            # pre-train modifications
            spks: torch.Tensor; cl: NDArray[np.int_]                     
            spks, cl = data
            if do_shft:
                targ: torch.Tensor = spks[:,:,:,5:-5].clone()

                # randomly time-shift 30% of input spikes
                shft_ind: NDArray[np.int_] = np.random.choice(np.arange(spks.shape[0]), int(spks.shape[0]*.3), replace=False)
                shft_spks: torch.Tensor = spks[:,:,:,5:-5].clone()
                for ind in shft_ind:
                    shift: NDArray[np.int_] = np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5], 1)[0]
                    shft_spks[ind,:,:,:] = spks[ind,:,:,5+shift:pre_samples+post_samples+5+shift].clone()
                spks = shft_spks
            else:
                targ = spks
                spks = spks

            # backprop
            out: torch.Tensor = net(spks)
            loss: torch.Tensor = loss_fn(out, targ)
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
        i: int = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                print("\rTesting batch " + str(i+1) + "/" + str(len(test_loader)), end="")

                # pre-test modifications
                spks, cl = data
                if do_shft:
                    targ = spks[:,:,:,5:-5].clone()

                    # randomly time-shift 30% of spikes
                    shft_ind: NDArray[np.int_] = np.random.choice(np.arange(spks.shape[0]), int(spks.shape[0]*.3), replace=False)
                    shft_spks: torch.Tensor = spks[:,:,:,5:-5].clone()
                    for ind in shft_ind:
                        shift: NDArray[np.int_] = np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5], 1)[0] # type: ignore
                        shft_spks[ind,:,:,:] = spks[ind,:,:,5+shift:pre_samples+post_samples+5+shift].clone().to(device)
                    spks = shft_spks
                else:
                    targ = spks
                    spks = spks
                targ[cl==-1,:,:,:] = 0  # noise snippets should be reconstructed as 0

                out = net(spks)
                tloss: torch.Tensor = loss_fn(out, targ)
                running_tloss += tloss

        avg_vloss: torch.Tensor | float = running_tloss/(i+1)
        test_loss[epoch] = avg_vloss

        print("\nLOSS | train: %.4f | test %.4f" % (last_loss, avg_vloss))
        
    return net, spk_data