from NiN_skin import nin_skin
from dataloaders import CustomDataLoader
from datasets import get_dataset
from metrics import BooleanMetrics, RunningAverage
from utils import getArgs, save_figure, log

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 
import matplotlib.pyplot as plt
import os 
import logging
from pathlib import Path

import argparse

# get settings as input arguments
train = False
args, dl_args = getArgs(train=train)

# save/load directories 
load_init = next(Path(f"./results/train/{args.load_init}").glob("*.pth"))
load_multi_path = next(Path(f"./results/train/{args.load_multi}").glob("*.pth")) if args.multi else None
save_path = f"./results/test/{args.fname}/{args.dataset_name}" 
LogPath = f"{save_path}/{args.fname}.txt"
ImageFolder = f"{save_path}/images/"
os.makedirs(Path(save_path).absolute(), exist_ok=True)
os.makedirs(Path(ImageFolder).absolute(), exist_ok=True)

# set up logging file 
logging.basicConfig(filename=LogPath, filemode='w', format='%(asctime)s - %(message)s', level=logging.ERROR)
logger = logging.getLogger()

# random seed 
random_seed = None
if random_seed:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

# get CPU/GPU device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

split_file = f"./Data/VidSequences/LIRIS/LIRIS_Data_{args.frame_num}_indices.npz" if args.dataset_name == "LIRIS" else None

# load model 
model = nin_skin(pretrained=True, load_path=load_init, load_multi_path=load_multi_path, **vars(args))

# load dataset 
dataset = get_dataset(**vars(args))
DL = CustomDataLoader(dataset, split_file=split_file, **dl_args)
test_loader = DL.getLoader(mode='test')
print(len(test_loader))

# loop over dataset
dataset.mode = "test"
model.eval()
torch.set_grad_enabled(False) # no gradient enabled

if args.local:

######### IMAGE MAX ###############

    running_F1, running_acc, running_prec, running_rec = RunningAverage(), RunningAverage(), RunningAverage(), RunningAverage()
    running_t = RunningAverage()
    for i, data in enumerate(test_loader, 0):

        # read in batch 
        inp, tgt = data.inp.to(device), data.tgt.to(device).bool()

        # forward step
        pred = model(inp)

        # interpolate
        if pred.dim() == 5:
            pred = pred.squeeze_(2)
        pred = F.interpolate(pred,size=tgt.squeeze().shape, mode='bilinear', align_corners=True)

        # choose threshold value that maximizes F1 score 
        best_F1, best_t = -float('inf'), 0
        for t in np.arange(pred.min(),pred.max(),0.01):

            pred_bin = pred > t

            # get F1 score 
            F1 = BooleanMetrics(pred_bin, tgt).F1()

            # maintain maximum 
            if F1 > best_F1:
                best_F1 = F1
                best_t = t 

        # get statistics for peak f-measure 
        pred_bin = pred > best_t

        M = BooleanMetrics(pred_bin, tgt)
        accuracy = M.Accuracy()
        precision = M.Precision()
        recall = M.Recall()
        F1 = M.F1()

        # running averages
        running_F1.add(F1)
        running_acc.add(accuracy)
        running_prec.add(precision)
        running_rec.add(recall)
        running_t.add(best_t)

        # # display batch statistics 
        # log(f"Output Min/Max: {pred.min():.2f}/{pred.max():.2f}", logger)
        # log(f"Threshold: {t:.2f}", logger)
        # log(f"Accuracy: {accuracy:.3f}", logger)  
        # log(f"Precision: {precision:.3f}", logger)  
        # log(f"Recall: {recall:.3f}", logger)  
        # log(f"F1 score: {F1:.3f}", logger) 
        # log("*************", logger) 
        
        save_figure(Path(ImageFolder) / f"{i}", inp, tgt, pred, pred_bin, 
                    grayscale=args.grayscale, show=False)

    log(f"Average Threshold: {running_t.avg:.2f}", logger)
    log(f"Average F1 score: {running_F1.avg:.4f}", logger)
    log(f"Average Accuracy: {running_acc.avg:.4f}", logger)
    log(f"Average Precision: {running_prec.avg:.4f}", logger)
    log(f"Average Recall: {running_rec.avg:.4f}", logger)

else:

########## DATSET MAX ###########

    # choose threshold value that maximizes F1 score 
    best_F1, best_t = -float('inf'), 0
    for t in np.arange(-0.5,0.5,0.01): 

        running_F1 = RunningAverage()
        for i, data in enumerate(test_loader, 0):
        
            # read in batch 
            inp, tgt = data.inp.to(device), data.tgt.to(device).bool()
            
            # forward step
            pred = model(inp)

            # interpolate
            if pred.dim() == 5:
                pred = pred.squeeze_(2)
            pred = F.interpolate(pred,size=tgt.squeeze().shape, mode='bilinear', align_corners=True)
            
            pred_bin = pred > t

            # get F1 score 
            F1 = BooleanMetrics(pred_bin, tgt).F1()

            # running averages
            running_F1.add(F1)

        # maintain maximum 
        if running_F1.avg > best_F1:
            best_F1 = running_F1.avg 
            best_t = t 

    running_F1, running_acc, running_prec, running_rec = RunningAverage(), RunningAverage(), RunningAverage(), RunningAverage()
    for i, data in enumerate(test_loader, 0):
        
        # read in batch 
        inp, tgt = data.inp.to(device), data.tgt.to(device).bool()
        
        # forward step
        pred = model(inp)

        # interpolate
        if pred.dim() == 5:
            pred = pred.squeeze_(2)
        pred = F.interpolate(pred,size=tgt.squeeze().shape, mode='bilinear', align_corners=True)

        # get statistics for peak f-measure 
        pred_bin = pred > best_t

        M = BooleanMetrics(pred_bin, tgt)
        accuracy = M.Accuracy()
        precision = M.Precision()
        recall = M.Recall()
        F1 = M.F1()

        # running averages
        running_F1.add(F1)
        running_acc.add(accuracy)
        running_prec.add(precision)
        running_rec.add(recall)

        save_figure(Path(ImageFolder) / f"{i}", inp, tgt, pred, pred_bin, 
                    grayscale=args.grayscale, show=False)

    log(f"Threshold: {best_t:.2f}", logger)
    log(f"Average F1 score: {running_F1.avg:.4f}", logger)
    log(f"Average Accuracy: {running_acc.avg:.4f}", logger)
    log(f"Average Precision: {running_prec.avg:.4f}", logger)
    log(f"Average Recall: {running_rec.avg:.4f}", logger)
