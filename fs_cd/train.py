from __future__ import print_function
import numpy as np
import json
import time
import sys
from datetime import datetime
import pathlib
import shutil
import yaml
from argparse import ArgumentParser
import os
from functools import partial
from sklearn import metrics
from tqdm import tqdm, trange
import torchvision.models as models

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from models.SUNet18 import SUNet18
from models.SiamUnet_conc import SiamUnet_conc
from models.Unet import Unet
from models.ResNet18 import ResNet18

from dataloader import Dataset
from augmentations import get_validation_augmentations, get_training_augmentations
from losses import choose_criterion2d,calMetric_iou
from optim import set_optimizer, set_scheduler
from cp import pretrain_strategy

def get_args():
    parser = ArgumentParser(description = "Hyperparameters", add_help = True)
    parser.add_argument('-c', '--config-name', type = str, help = 'YAML Config name', dest = 'CONFIG', default = 'SECOND')
    parser.add_argument('-sn','--savename', type = str, dest = 'savename', default = 'four')
    parser.add_argument('-nw', '--num-workers', type = str, help = 'Number of workers', dest = 'num_workers', default = 2)
    parser.add_argument('-g', '--gpu', type = int, help = 'Selected gpu', dest = 'gpu', default = 3)
    return parser.parse_args() 

# to calculate rmse
def metric_mse(inputs, targets, exclude_zeros = False):
    loss = (inputs - targets) ** 2
    if exclude_zeros:
        n_pixels = np.count_nonzero(targets)
        return np.sum(loss)/n_pixels
    else:
        return np.mean(loss)

args = get_args()

device = 'cuda'
cuda = True
num_GPU = 1
torch.cuda.set_device(args.gpu)
manual_seed = 18
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

config_name = args.CONFIG
config_path = '/media/lthpc/hd_auto/Liu/caozhijuan/Few_Shot_CD/fs_cd/config/'+config_name
default_dst_dir = "./results/" + args.savename
out_file = default_dst_dir + config_name + '/'
os.makedirs(out_file, exist_ok=True)

# Load the configuration params of the experiment
full_config_path = config_path + ".yaml"
print(f"Loading experiment {full_config_path}")
with open(full_config_path, "r") as f:
    exp_config = yaml.load(f, Loader=yaml.SafeLoader)

print(f"Logs and/or checkpoints will be stored on {out_file}")
shutil.copyfile(full_config_path, out_file+'config.yaml')
print("Config file correctly saved!")

stats_file = open(out_file + 'stats.txt', 'a', buffering=1)
print(' '.join(sys.argv), file=stats_file)
print(' '.join(sys.argv))

print(exp_config)
print(exp_config, file=stats_file)

dir = exp_config['data']['train']['path']

batch_size = exp_config['data']['train']['batch_size']

lweight2d, lweight3d = exp_config['model']['loss_weights']
weights2d = exp_config['model']['2d_loss_weights']

augmentation = exp_config['data']['augmentations']

mean = exp_config['data']['mean']
std = exp_config['data']['std']
dataset_name = exp_config['data']['dataset_name']

if augmentation:
    train_transform = get_training_augmentations(m = mean, s = std)
else:
  train_transform = get_validation_augmentations(m = mean, s = std)

valid_transform = get_validation_augmentations(m = mean, s = std)

train_dataset = Dataset(dir,
                        augmentation = train_transform,
                        dataset_name = dataset_name,
                        phase='train')

valid_dataset = Dataset(dir,
                        augmentation = valid_transform,
                        dataset_name = dataset_name,
                        phase='test')
                        
test_dataset = Dataset(dir,
                        augmentation = valid_transform,
                        dataset_name = dataset_name,
                        phase='test')


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)


class_weights2d = torch.FloatTensor(weights2d).to(device)
name_2dloss = exp_config['model']['2d_loss'] 
criterion2d = choose_criterion2d(name_2dloss, class_weights2d) #, class_ignored)
criterion = nn.CosineSimilarity(dim=1).to(device)

nepochs = exp_config['optim']['num_epochs']
lr = exp_config['optim']['lr']

model = exp_config['model']['model']
classes = exp_config['model']['num_classes']

pretrain = exp_config['model']['pretraining_strategy']
arch = exp_config['model']['feature_extractor_arch']
CHECKPOINTS = exp_config['model']['checkpoints_path']

encoder, pretrained, _ = pretrain_strategy(pretrain, CHECKPOINTS, arch)

if model == "SUNet18":
    net = SUNet18(3, 2, share_encoder = False, base_model = encoder).to(device)
    # net = torch.load('/home/orange/ZCJ/3DCD_test/results/example_sunet_training/2dbestnet.pth')
elif model == "ResNet18":
    net = ResNet18(3, 2, share_encoder = False, base_model = encoder).to(device)
elif model == "SiamUnet_conc":
	net = SiamUnet_conc(3,2).to(device)
elif model == "Unet":
	net = Unet(3,2).to(device)
else:
	print('Model not implemented yet')
print('Model selected: ', model)

optimizer = set_optimizer(exp_config['optim'], net)
print('Optimizer selected: ', exp_config['optim']['optim_type'])
lr_adjust = set_scheduler(exp_config['optim'], optimizer)
print('Scheduler selected: ', exp_config['optim']['lr_schedule_type'])

res_cp = exp_config['model']['restore_checkpoints']
if os.path.exists(out_file+f'{res_cp}bestnet.pth'):
  net.load_state_dict(torch.load(out_file+f'{res_cp}bestnet.pth'))
  print('Checkpoints successfully loaded!')
else:
  print('No checkpoints founded')

start = time.time()

best2dmetric = 0
  
net.train()

for epoch in range(1, nepochs):
  tot_2d_loss = 0

  for param_group in optimizer.param_groups:
    print("Epoch: %s" % epoch, " - Learning rate: ", param_group['lr'])

  for st1, st2, qt1, qt2, smask, qmask in train_loader:

    st1 = st1.to(device)
    st2 = st2.to(device)

    qt1 = qt1.to(device)
    qt2 = qt2.to(device)

    smask = smask.to(device)
    qmask = qmask.to(device)

    optimizer.zero_grad()

    out2d = net(st1, st2, smask, qt1, qt2)
    loss = criterion2d(out2d, qmask.long()) #long 

    loss.backward() #bacward delle loss
    optimizer.step()


    tot_2d_loss += loss.detach().cpu().numpy()*batch_size 

  epoch_2d_loss = tot_2d_loss/len(train_dataset)
  epoch_loss = epoch_2d_loss
  
  lr_adjust.step()
  
  print(f"Training loss: {epoch_loss},\t2D Loss: {epoch_2d_loss},")

  with torch.no_grad():
    net.eval()

    TN = 0
    FP = 0
    FN = 0
    TP = 0

    for st1, st2, qt1, qt2, smask, qmask in valid_loader:

      st1 = st1.to(device)
      st2 = st2.to(device)

      qt1 = qt1.to(device)
      qt2 = qt2.to(device)

      smask = smask.to(device)
      qmask = qmask.to(device)

      out2d = net(st1, st2, smask, qt1, qt2)
      out2d = out2d.detach().argmax(dim=1)
   
      # try:
      #     tn, fp, fn, tp = metrics.confusion_matrix(qmask.ravel(), out2d.ravel()).ravel()
      # except: 
      #     tn, fp, fn, tp = [0,0,0,0]
      #     #print('Only 0 mask') 
      tp,fp,tn,fn = calMetric_iou(out2d,qmask)

      TN += tn
      FP += fp
      FN += fn 
      TP += tp

    
    F1 = 2*TP/(2*TP+FN+FP)
    IoU = TP/(TP+FN+FP)
    ACC = (TP+TN)/(TP+TN+FP+FN)

    
    print(f'Validation metrics - 2D: F1 Score -> {F1*100} %; mIoU -> {IoU*100} %; ACC -> {ACC*100} %')

    if F1 > best2dmetric:
      best2dmetric = F1
      torch.save(net, out_file+'/best_'+dataset_name+'.pth')
      print('Best 2D model saved!')


  stats = dict(epoch = epoch, Loss2D = epoch_2d_loss, Loss = epoch_loss, F1Score = F1*100, IoU = IoU*100)


end = time.time()
print('Training completed. Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
print(f'Best metrics: F1 score -> {best2dmetric*100} %')

start = time.time()

if os.path.exists('%s/' % out_file + f'{res_cp}bestnet.pth'):
    net.load_state_dict(torch.load('%s/' % out_file + f'{res_cp}bestnet.pth'))
    print("Checkpoints correctly loaded: ", out_file)

net.eval()

TN = 0
FP = 0
FN = 0
TP = 0


for st1, st2, qt1, qt2, smask, qmask in test_loader:
  st1 = st1.to(device)
  st2 = st2.to(device)

  qt1 = qt1.to(device)
  qt2 = qt2.to(device)

  smask = smask.to(device)
  qmask = qmask.to(device)
      
  out2d = net(st1, st2, smask, qt1, qt2)
  out2d = out2d.detach().argmax(dim=1)

  # try:
  #     tn, fp, fn, tp = metrics.confusion_matrix(qmask.ravel(), out2d.ravel()).ravel()
  # except: 
  #     tn, fp, fn, tp = [0,0,0,0]
  #     #print('Only 0 mask') 
  tp,fp,tn,fn = calMetric_iou(out2d.ravel(),qmask.ravel())

  TN += tn
  FP += fp
  FN += fn 
  TP += tp


F1 = 2*TP/(2*TP+FN+FP)
IoU = TP/(TP+FN+FP)
ACC = (TP+TN)/(TP+TN+FP+FN)


end = time.time()
print('Test completed. Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
print(f'Test metrics - 2D: F1 Score -> {F1*100} %; mIoU -> {IoU*100} %; ACC -> {ACC*100} %')
stats = dict(epoch = 'Test', F1Score = F1*100, IoU = IoU*100, ACC = ACC*100)

