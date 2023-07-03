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
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split


def generate_train_data(data, times_multi, clusters, counts, labels, mean_wf, channel_pos, folder):
    chans = {}

    for i in range(mean_wf.shape[0]):
        if i in counts:
            chs, peak = bd.utils.find_best_channels(mean_wf[i], channel_pos)
            dists =  bd.utils.get_dists(channel_pos, peak, chs)
            chans[i] = chs[np.argsort(dists)].tolist()
            
    file_names = []
    cl_ids = []

    pre_samples = 10
    post_samples = 30

    max_spikes = 2000
    min_spikes = 100
    
    tot = 0
    for i in range(clusters.max()+1):
        if (i in counts) and (counts[i] > min_spikes) and (labels.loc[labels['cluster_id']==i, 'group'].item() == 'good'):
            cl_times = times_multi[i].astype("int32")

            # cap number of spikes
            if (max_spikes < cl_times.shape[0]):
                np.random.shuffle(cl_times)
                cl_times = cl_times[:max_spikes]

            # save spikes to file and add to annotations
            for j in range(cl_times.shape[0]):
                file_names.append("cl%d_spk%d.npy" % (i, j))
                cl_ids.append(i)

                spike = data[cl_times[j]-pre_samples:cl_times[j]+post_samples, chans[i]].T
                spike = np.nan_to_num(spike)
                out_name = "./data/" + folder +"/cl%d_spk%d.npy" % (i, j)

                np.save(out_name, spike)
                    
        df = pd.DataFrame({'file':file_names, 'cl':cl_ids}, index=None)
        df.to_csv("./data/" + folder + "/labels.csv", header=False, index=False)
        
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
    
"""
A Convolutional Autoencoder
"""
class CN_AE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=256*1*5, zDim=15):
        super(CN_AE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encUpSamp1 = nn.Upsample((16,80))
        self.encUpConv1 = nn.ConvTranspose2d(imgChannels, 16, (3,3), padding=(1,1))
        
        self.encDownConv1 = nn.Conv2d(16, 32, (3,3), padding='same')
        self.encDownPool1 = nn.MaxPool2d((2,2))
        
        self.encConv1 = nn.Conv2d(32, 64, (3,3), padding='same')
        self.encBN1 = nn.BatchNorm2d(64)
        self.encPool1 = nn.MaxPool2d((2,2), return_indices=True)
        
        self.encConv2 = nn.Conv2d(64, 128, (3,3), padding='same')
        self.encBN2 = nn.BatchNorm2d(128)
        self.encPool2 = nn.MaxPool2d((2,2), return_indices=True)
        
        self.encConv3 = nn.Conv2d(128, 256, (3,3), padding='same')
        self.encBN3 = nn.BatchNorm2d(256)
        self.encPool3 = nn.MaxPool2d((2,2), return_indices=True)
        
        self.encFC1 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decBN1 = nn.BatchNorm2d(256)
        
        self.decUpSamp1 = nn.Upsample((2,10))
        self.decConv1 = nn.ConvTranspose2d(256, 128, (3,3), padding=(1,1))
        self.decBN2 = nn.BatchNorm2d(128)
        
        self.decUpSamp2 = nn.Upsample((4,20))
        self.decConv2 = nn.ConvTranspose2d(128, 64, (3,3), padding=(1,1))
        self.decBN3 = nn.BatchNorm2d(64)
        
        self.decUpSamp3 = nn.Upsample((8,40))
        self.decConv3 = nn.ConvTranspose2d(64, 32, (3,3), padding=(1,1))
        
        self.decUpSamp4 = nn.Upsample((16, 80))
        self.decUpConv1 = nn.ConvTranspose2d(32, 16, (3,3), padding=(1,1))
        
        self.decDownConv1 = nn.Conv2d(16, imgChannels, (3,3), padding='same')
        self.decPool1 = nn.MaxPool2d((2,2))

    def encoder(self, x):
        # Input is fed into 2 convolutional layers sequentially
        x = self.encUpSamp1(x)
        x = F.relu(self.encUpConv1(x))
        
        x = F.relu(self.encDownConv1(x))
        x = self.encDownPool1(x)
        
        x = F.relu(self.encBN1(self.encConv1(x)))
        x, self.encInd1 = self.encPool1(x)
        
        x = F.relu(self.encBN2(self.encConv2(x)))
        x, self.encInd2 = self.encPool2(x)
        
        x = F.relu6(self.encBN3(self.encConv3(x)))
        x, self.encInd3 = self.encPool3(x)
        
        x = x.view(-1, 256*1*5)
        
        x = self.encFC1(x)
        
        return x

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = self.decFC1(z)
        x = x.view(-1, 256, 1, 5)
        x = F.relu(self.decBN1(x))
        
        x = self.decUpSamp1(x)
        x = F.relu(self.decBN2(self.decConv1(x)))
        
        x = self.decUpSamp2(x)
        x = F.relu(self.decBN3(self.decConv2(x)))
        
        x = self.decUpSamp3(x)
        x = F.relu(self.decConv3(x))

        x = self.decUpSamp4(x)
        x = F.relu(self.decUpConv1(x))
        
        x = self.decDownConv1(x)
        x = self.decPool1(x)
        
        return x

    def forward(self, x):

        # The entire pipeline of the AE: encoder -> decoder
        z = self.encoder(x)
        out = self.decoder(z)
        
        return out
        
def train_ae(folder, num_epochs, zDim=15):
    spk_data = SpikeDataset("./data/"+ folder +"/labels.csv","./data/"+ folder +"/", ToTensor())
    
    BATCH_SIZE = 128

    labels = spk_data.spk_labels.iloc[:, 1]

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices, _, _ = train_test_split(
        range(len(spk_data)),
        labels,
        stratify=labels,
        test_size=0.2,
        random_state=42
    )

    # generate subset based on indices
    train_split = Subset(spk_data, train_indices)
    test_split = Subset(spk_data, test_indices)

    # create batches
    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_split, batch_size=BATCH_SIZE)
    
    """
    Determine if any GPUs are available
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    """
    Initialize Hyperparameters
    """
    learning_rate = 1e-3
    test_loss = np.zeros(num_epochs)

    """
    Initialize the network and the Adam optimizer
    """
    net = CN_AE(zDim=zDim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    """
    Training the network for a given number of epochs
    The loss after every epoch is printed
    """
    for epoch in range(num_epochs):
        print('EPOCH %d' % (epoch))
        running_loss = 0
        last_loss = 0
        for idx, data in enumerate(train_loader, 0):
            print("\r" + str(idx), end="")
            imgs, _ = data
            imgs = imgs.to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out = net(imgs)
            loss = loss_fn(out, imgs)


            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_sched.step()

            # Gather data and report
            running_loss += loss
            if idx % 500 == 499:
                last_loss = running_loss/500
                print(" Batch %d | loss: %.4f" % (idx, last_loss))
                running_loss = 0

        print()
        # test error
        running_tloss = 0
        net.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                print("\r" + str(i), end="")
                imgs, _ = data
                imgs = imgs.to(device)

                out = net(imgs)
                tloss = loss_fn(out, imgs)
                running_tloss += tloss

        avg_vloss = running_tloss/(i+1)
        test_loss[epoch] = avg_vloss

        print("\nLOSS | train: %.4f | test %.4f" % (last_loss, avg_vloss))
        
    return net, spk_data