
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
opt = torch.optim.Adam(segnet.parameters(), lr=0.01)
train_loss, valid_loss = train(segnet, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=5)

torch.save(train_loss, "/content/cloud_segmenter/metrics/train_loss")
torch.save(valid_loss, "/content/cloud_segmenter/metrics/valid_loss")