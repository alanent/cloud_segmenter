

import matplotlib.pyplot as plt
import torch
import time
from torch.utils.data import Dataset, DataLoader, sampler
from IPython.display import clear_output
from torch import nn
import numpy as np
import pandas as pd
from torchmetrics import JaccardIndex


class SimpleCNN(nn.Module):
    
    def __init__(self):
        super().__init__() 
        
        self.encod = nn.Sequential(
            nn.Conv2d(4,32, kernel_size=5, stride=1, padding=2), #out = 32x384x384 cambie 16 por 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #out = 32x192x192, acum = 1/2

            nn.Conv2d(32,64, kernel_size=5, stride=1, padding=2), #out = 64x192x192
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #out = 64x96x96, acum = 1/4
        
        
            nn.Conv2d(64,128, kernel_size=5, stride=1, padding=2), #out=128x96x96
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #out = 128x48x48, acum = 1/8
        
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2), #out = 256x48x48
            nn.BatchNorm2d(256),
            nn.ReLU(),    
            nn.MaxPool2d(kernel_size=2, stride=2), #out = 256x24x24, acum = 1/16
        
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2), #out = 256x24x24
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #out = 512x12x12, acum = 1/32
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), #out = 1024x24x24
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #out = 1024x6x6, acum = 1/64
            
            nn.Conv2d(1024, 1024, kernel_size=5, stride=1, padding=2), #out = 2048x12x12
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) #out = 1024x3x3, acum = 1/128
            
        )
        
        self.decod = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=3,stride=1, padding=2), #out = 512x3x3
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.UpsamplingNearest2d([6,6], scale_factor=None), #out=128x6x6 
            
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3,stride=1, padding=2), #out = 256x6x6
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.UpsamplingNearest2d([12,12], scale_factor=None), #out=128x6x6 
            
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=5,stride=1, padding=2), #out=128x12x12
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.UpsamplingNearest2d([24,24], scale_factor=None), #out=128x24x24
        
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=5,stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.UpsamplingNearest2d([48,48], scale_factor=None), #out=64x48x48
           
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=5,stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.UpsamplingNearest2d([96,96], scale_factor=None), #out=32x96x96
        
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=5,stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.UpsamplingNearest2d([192,192], scale_factor=None),#out=16x192x192
        
            nn.ConvTranspose2d(in_channels=16,out_channels=2,kernel_size=5,stride=1, padding=2),
            nn.BatchNorm2d(2),
            nn.Sigmoid(), 
            nn.UpsamplingNearest2d([384,384], scale_factor=None),#out=4x384x384
        )
            
    
    def forward(self, xb):
        out = self.encod(xb) #out = 512x12x12
        out = self.decod(out)
        return out

cnn=SimpleCNN()

#LOAD STATE DICT 
cnn = torch.load('/content/cloud_segmenter/models/model.pt')

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
        predb = cnn(xb.cuda())

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