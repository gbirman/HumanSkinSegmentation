from NiN_skin import nin_skin
from dataloaders import CustomDataLoader
from datasets import get_dataset
from metrics import RunningAverage
from utils import getArgs, log

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

import os
import sys
import logging
from pathlib import Path

# get settings as input arguments
train = True
args, dl_args = getArgs(train=train)

# save/load directories  
load_init = next(Path(f"./results/train/{args.load_init}").glob("*.pth"))
save_path = f"./results/train/{args.fname}/"
PATH = f"{save_path}/{args.fname}.pth"
LogPath = f"{save_path}/{args.fname}.txt"
LossPath = f"{save_path}/{args.fname}.npz"
os.makedirs(Path(save_path).absolute(), exist_ok=True)

# set up logging file 
logging.basicConfig(filename=LogPath, filemode='w', format="%(asctime)s;%(message)s", level=logging.ERROR)
logger = logging.getLogger()

split_file = f"./Data/VidSequences/LIRIS/LIRIS_Data_{args.frame_num}_indices.npz" if args.dataset_name == "LIRIS" else None

# random seed 
random_seed = None
if random_seed:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

# get CPU/GPU device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# instantiate model 
model = nin_skin(pretrained=False, load_path=load_init, **vars(args))

# get loss function
criterion = nn.MSELoss() if args.original else nn.BCEWithLogitsLoss()

# get optimizer parameters
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

# load dataset
dataset = get_dataset(**vars(args))
DL = CustomDataLoader(dataset, split_file=split_file, **dl_args)
train_loader = DL.getLoader(mode='train')
val_loader = DL.getLoader(mode='val') 

num_epochs = 500
best_val, best_epoch = float('inf'), 0 # current best validation loss
val_flag = True # true if validating
val_stop_iters_improvement = 25 # keep going this many iterations when validation loss not improving
bad_iters_no_improvement = 0
torch.set_grad_enabled(True) # train with gradient enabled
training_loss, validation_loss = [], []
for epoch in range(num_epochs):  # loop over the dataset multiple times

    dataset.mode = "train"
    model.train()
    running_loss = RunningAverage()
    for i, data in enumerate(train_loader, 0):
        
        # read in batch 
        inp, tgt = data.inp.to(device), data.tgt.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = model(inp)
        loss = criterion(pred, tgt)
        # log(f"{loss:.3f} ({i+1}/{len(train_loader)})", logger)
        loss.backward()
        optimizer.step()

        # running average
        running_loss.add(loss.item())
        
    # maintain avg training loss 
    training_loss = np.append(training_loss, running_loss.avg)

    # log(f"TRAIN -- epoch: {epoch+1}, loss: {running_loss.avg:.3f}", logger)

    if val_flag:
        dataset.mode = "val"
        model.eval()
        running_loss = RunningAverage()
        for i, data in enumerate(val_loader, 0):
            
            # read in batch 
            inp, tgt = data.inp.to(device), data.tgt.to(device).float()

            with torch.no_grad():
                # forward + backward + optimize
                pred = model(inp)
                loss = criterion(pred, tgt)
            
            # log(f"{loss:.3f} ({i+1}/{len(val_loader)})", logger)

            # running average
            running_loss.add(loss.item())

        # log(f"EVAL -- epoch: {epoch+1}, loss: {running_loss.avg:.3f}", logger)

        # maintain avg validation loss 
        validation_loss = np.append(validation_loss, running_loss.avg)

        # learning rate scheduler 
        scheduler.step(running_loss.avg)

        # keep track of best model params 
        # NOTE: modify to save at end 
        if running_loss.avg < best_val:
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), PATH)
            else:
                torch.save(model.state_dict(), PATH)
            best_val = running_loss.avg
            best_epoch = epoch+1

        # log(f"BEST -- epoch: {best_epoch}, loss: {best_val:.3f}", logger)

        # log all useful info in one line 
        log(f"Epoch {epoch+1} -- train: {training_loss[-1]:.3f}, eval: {validation_loss[-1]:.3f}, best: {best_val:.3f} ({best_epoch})", logger)

        # exit conditions 
        if running_loss.avg > best_val:

            # exit if no improvement shown for val_stop_iters_improvement
            bad_iters_no_improvement += 1 
            if bad_iters_no_improvement == val_stop_iters_improvement:
                break

        else:
            bad_iters_no_improvement = 0

# save training loss 
np.savez(LossPath, train=training_loss, val=validation_loss)
