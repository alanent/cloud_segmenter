

import matplotlib.pyplot as plt
import torch
import time
from torch.utils.data import Dataset, DataLoader, sampler
from IPython.display import clear_output
from torch import nn
import numpy as np
import pandas as pd
from torchmetrics import JaccardIndex




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os

#Training settings
parser = argparse.ArgumentParser(description='PyTorch SegNet example')
parser.add_argument('--batch-size', type=int, default=12, metavar='N', help='training batch-size (default: 12)')
parser.add_argument('--epochs', type=int, default=300, metavar='E', help='no. of epochs to run (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='MOM', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--in-chn', type=int, default=3, metavar='IN', help='input image channels (default: 3 (RGB Image))')
parser.add_argument('--out-chn', type=int, default=32, metavar='OUT', help='output channels/semantic classes (default: 32)')

hyperparam = parser.parse_args()

hyperparam.cuda = not hyperparam.no_cuda and torch.cuda.is_available()
USE_CUDA = hyperparam.cuda

class SegNet(nn.Module):

	def __init__(self, BN_momentum=hyperparam.momentum):
		super(SegNet, self).__init__()

		#SegNet Architecture
		#Takes input of size in_chn = 4 (RGB images have 3 channels)
		#Outputs size label_chn (N # of classes)

		#ENCODING consists of 5 stages
		#Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
		#Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

		#General Max Pool 2D for ENCODING layers
		#Pooling indices are stored for Upsampling in DECODING layers

		self.in_chn = hyperparam.in_chn
		self.out_chn = hyperparam.out_chn

		self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 

		self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
		self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
		self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

		self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
		self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

		self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
		self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
		self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

		self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

		self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)


		#DECODING consists of 5 stages
		#Each stage corresponds to their respective counterparts in ENCODING

		#General Max Pool 2D/Upsampling for DECODING layers
		self.MaxDe = nn.MaxUnpool2d(2, stride=2) 

		self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)

		self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
		self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)

		self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
		self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
		self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
		self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

		self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
		self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
		self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

		self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
		self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
		self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

	def forward(self, x):

		#ENCODE LAYERS
		#Stage 1
		x = F.relu(self.BNEn11(self.ConvEn11(x))) 
		x = F.relu(self.BNEn12(self.ConvEn12(x))) 
		x, ind1 = self.MaxEn(x)
		size1 = x.size()

		#Stage 2
		x = F.relu(self.BNEn21(self.ConvEn21(x))) 
		x = F.relu(self.BNEn22(self.ConvEn22(x))) 
		x, ind2 = self.MaxEn(x)
		size2 = x.size()

		#Stage 3
		x = F.relu(self.BNEn31(self.ConvEn31(x))) 
		x = F.relu(self.BNEn32(self.ConvEn32(x))) 
		x = F.relu(self.BNEn33(self.ConvEn33(x))) 	
		x, ind3 = self.MaxEn(x)
		size3 = x.size()

		#Stage 4
		x = F.relu(self.BNEn41(self.ConvEn41(x))) 
		x = F.relu(self.BNEn42(self.ConvEn42(x))) 
		x = F.relu(self.BNEn43(self.ConvEn43(x))) 	
		x, ind4 = self.MaxEn(x)
		size4 = x.size()

		#Stage 5
		x = F.relu(self.BNEn51(self.ConvEn51(x))) 
		x = F.relu(self.BNEn52(self.ConvEn52(x))) 
		x = F.relu(self.BNEn53(self.ConvEn53(x))) 	
		x, ind5 = self.MaxEn(x)
		size5 = x.size()

		#DECODE LAYERS
		#Stage 5
		x = self.MaxDe(x, ind5, output_size=size4)
		x = F.relu(self.BNDe53(self.ConvDe53(x)))
		x = F.relu(self.BNDe52(self.ConvDe52(x)))
		x = F.relu(self.BNDe51(self.ConvDe51(x)))

		#Stage 4
		x = self.MaxDe(x, ind4, output_size=size3)
		x = F.relu(self.BNDe43(self.ConvDe43(x)))
		x = F.relu(self.BNDe42(self.ConvDe42(x)))
		x = F.relu(self.BNDe41(self.ConvDe41(x)))

		#Stage 3
		x = self.MaxDe(x, ind3, output_size=size2)
		x = F.relu(self.BNDe33(self.ConvDe33(x)))
		x = F.relu(self.BNDe32(self.ConvDe32(x)))
		x = F.relu(self.BNDe31(self.ConvDe31(x)))

		#Stage 2
		x = self.MaxDe(x, ind2, output_size=size1)
		x = F.relu(self.BNDe22(self.ConvDe22(x)))
		x = F.relu(self.BNDe21(self.ConvDe21(x)))

		#Stage 1
		x = self.MaxDe(x, ind1)
		x = F.relu(self.BNDe12(self.ConvDe12(x)))
		x = self.ConvDe11(x)

		return x


segnet=SegNet()

#LOAD STATE DICT 
segnet = torch.load('/content/cloud_segmenter/models/model.pt')

#Load validation dataset
valid_ds = torch.load("/content/cloud_segmenter/data/data_processed/valid_ds")
valid_dl = DataLoader(valid_ds, batch_size=24, shuffle=False)



#BATCH SIZE
bs = 12

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()



jaccard = JaccardIndex(num_classes=2,average='weighted')
jaccard_results = pd.DataFrame()


for i in range(200):
    xb, yb = next(iter(valid_dl))
    with torch.no_grad():
        predb = segnet(xb.cuda())

    for i in range(bs):
        jaccard_results = jaccard_results.append({'jaccard_index':jaccard(predb_to_mask(predb, i), yb[i]).numpy()}, ignore_index=True)


import json
with open('/content/cloud_segmenter/metrics/test_metrics.json', 'w') as f:
    json.dump({'jaccard_index':jaccard_results.jaccard_index.mean()}, f)


fig, ax = plt.subplots(bs,3, figsize=(12,bs*3.5))
for i in range(bs):
    ax[i,0].set_title('image')
    ax[i,0].imshow(batch_to_img(xb,i))
    ax[i,1].set_title('true mask')
    ax[i,1].imshow(yb[i])
    ax[i,2].set_title('prediction, jaccard-index = '+str(jaccard(predb_to_mask(predb, i), yb[i])))
    ax[i,2].imshow(predb_to_mask(predb, i))

fig.savefig('/content/cloud_segmenter/output/predictions.png')