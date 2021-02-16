import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time

from networks.dinknet import DinkNet34, DinkNet34_SP
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder


NAME = 'spdlinknet34_d_new'
solverIn = DinkNet34_SP

BATCHSIZE_PER_CARD = 4


ROOT = 'dataset/train/'
imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
trainlist = list(map(lambda x: x[:-8], imagelist))

SHAPE = (1024,1024)
solver = MyFrame(solverIn, dice_bce_loss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4)

mylog = open('logs/'+NAME+'.log','w')
tic = time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100.

for epoch in range(1, total_epoch + 1):
    if epoch > 120 and epoch % 5 == 0:
        solver.save('weights/' + NAME + '_epoch' + str(epoch) + '.th')
    
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    print('********', file=mylog)
    print('epoch:',epoch,'    time:',int(time()-tic), file=mylog)
    print('train_loss:',train_epoch_loss, file=mylog)
    print('SHAPE:',SHAPE, file=mylog)
    print('********')
    print('epoch:',epoch,'    time:',int(time()-tic))
    print('train_loss:',train_epoch_loss)
    print('SHAPE:',SHAPE)
    
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/'+NAME+'.th')
    if no_optim > 6:
        print('early stop at %d epoch' % epoch, file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
    mylog.flush()
    
print('Finish!', file=mylog)
print('Finish!')
mylog.close()