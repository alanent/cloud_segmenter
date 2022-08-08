
from torch.utils.data import Dataset, DataLoader
from prepare import CloudDataset
import os
import torch
import matplotlib.pyplot as plt
from pprint import pprint
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn


train_ds = torch.load("/content/cloud_segmenter/data/data_processed/train_ds", "train")
valid_ds = torch.load("/content/cloud_segmenter/data/data_processed/valid_ds", "valid")

train_dl = DataLoader(train_ds, batch_size=16, shuffle=False)
valid_dl = DataLoader(valid_ds,  batch_size=16, shuffle=False)


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



import time
from IPython.display import clear_output

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.cuda()
                y = y.cuda()
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 

                if step % 100 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            clear_output(wait=True)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    torch.save(model, "/content/cloud_segmenter/models/model.pt")
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss    


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()


loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(cnn.parameters(), lr=0.01)
train_loss, valid_loss = train(cnn, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=5)

torch.save(train_loss, "/content/cloud_segmenter/metrics/train_loss")
torch.save(valid_loss, "/content/cloud_segmenter/metrics/valid_loss")