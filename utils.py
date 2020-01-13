
from pathlib import Path
import os
from distutils.dir_util import copy_tree 

def getArgs(train=True):
    import argparse

    # determine if training on local or remote
    parser = argparse.ArgumentParser()
    parser.add_argument('-lo', '--local', action="store_true", dest="local")
    args = parser.parse_known_args()[0]

    # default settings
    fname = "BGR"
    data_root="./Data" # where is the data stored 
    just_subtract = False
    lr = 1e-3
    wd = 2e-4
    frame_num = 5
    load_init = "BGR_original"
    load_multi = "LIRIS_5_res_wd"
    if train:
        dataset_name = "FSD"
    else:
        dataset_name = "Pratheepan_Face"

    # dataloader settings  
    num_workers = 8 if args.local else 32
    pin_memory = False if args.local else True
    if train:
        batch_size = 8 if args.local else 32 
        validation_split = 0.3
        shuffle_dataset = True 
    else:
        batch_size = 1 
        validation_split = 0.3
        shuffle_dataset = False 

    # optional command line arguments
    parser.add_argument('-s', '--sigmoid', action="store_true", dest="use_sigmoid")
    parser.add_argument('-g', '--grayscale', action="store_true", dest="grayscale")
    parser.add_argument('-lr', action="store", dest="lr", default=lr, type=float)
    parser.add_argument('-fn', '--file_name', action="store", dest="fname", default=fname, type=str)
    parser.add_argument('-dr', '--data_root', action="store", dest="data_root", default=data_root, type=str)
    parser.add_argument('-dn', '--dataset_name', action="store", dest="dataset_name", default=dataset_name, type=str)
    parser.add_argument('-c', '--create_data_file', action="store_true", dest="createDataFile")
    parser.add_argument('-js', '--just_subtract', action="store_true", dest="just_subtract", default=just_subtract)
    parser.add_argument('-bs', '--batch_size', action="store", dest="batch_size", default=batch_size, type=int)
    parser.add_argument('-ln', '--load_init', action="store", dest="load_init", default=load_init, type=str)
    parser.add_argument('-m', '--multi', action="store_true", dest="multi")
    parser.add_argument('-fr', '--frame_num', action="store", dest="frame_num", default=frame_num, type=int)
    parser.add_argument('-lm', '--load_multi', action="store", dest="load_multi", default=load_multi, type=str)
    parser.add_argument('-o', '--original', action="store_true", dest="original")
    parser.add_argument('-r', '--res', action="store_true", dest="res")
    parser.add_argument('-wd', '--weight_decay', action="store", dest="wd", default=wd, type=float)
    args = parser.parse_args()
    dl_args = {'batch_size': args.batch_size, 'validation_split': validation_split, 'shuffle_dataset': shuffle_dataset,
                'pin_memory': pin_memory, 'num_workers': num_workers}
    print(vars(args), dl_args)

    return args, dl_args

# display figures for evaluation purposes (assumes batch_size = 1)
def save_figure(save_file, inp, tgt, pred, pred_bin, grayscale=False, show=False):
    import matplotlib.pyplot as plt 

    # bring to shape: N x C x D x H X W 
    if inp.dim() == 4: 
        inp = inp.unsqueeze_(2)

    if tgt.dim() == 4:
        tgt = tgt.unsqueeze_(2)

    if pred.dim() == 4:
        pred = pred.unsqueeze_(2)

    if pred_bin.dim() == 4:
        pred_bin = pred_bin.unsqueeze_(2)

    # bring to shape: C x D x H X W 
    inp = inp.squeeze(0)
    tgt = tgt.squeeze(0)
    pred = pred.squeeze(0)
    pred_bin = pred_bin.squeeze(0)

    # show total 
    _, axs = plt.subplots(inp.shape[1], 4, frameon=False) # D x 4 
    if inp.shape[1] == 1:
        axs = [axs]
    axs[0][0].set_title('Input')
    axs[0][1].set_title('Target')
    axs[0][2].set_title('Predicted')
    axs[0][3].set_title('Binarized')
    for i in range(inp.shape[1]):
        inp_img = inp[:,i,:,:].permute(1,2,0).flip(-1).squeeze(2).cpu().numpy()
        axs[i][0].imshow(inp_img)
        if i == inp.shape[1]//2:
            tgt_img = tgt[:,0,:,:].permute(1,2,0).squeeze(2).cpu().numpy()
            pred_img = pred[:,0,:,:].permute(1,2,0).squeeze(2).cpu().numpy()
            pred_bin_img = pred_bin[:,0,:,:].permute(1,2,0).squeeze(2).cpu().numpy()
            axs[i][1].imshow(tgt_img)
            axs[i][2].imshow(pred_img)
            axs[i][3].imshow(pred_bin_img)
        axs[i][0].axis('off')
        axs[i][1].axis('off')
        axs[i][2].axis('off')
        axs[i][3].axis('off')
    plt.savefig(f"{save_file}.png")
    if show:
        plt.show()
    plt.close()

    # show individual
    for i in range(inp.shape[1]):
        inp_img = inp[:,i,:,:].permute(1,2,0).flip(-1).squeeze(2).cpu().numpy()
        plt.imshow(inp_img)
        plt.axis('off')
        plt.savefig(f"{save_file}inp_{i}.png")
        plt.close()
        if i == inp.shape[1]//2:
            tgt_img = tgt[:,0,:,:].permute(1,2,0).squeeze(2).cpu().numpy()
            pred_img = pred[:,0,:,:].permute(1,2,0).squeeze(2).cpu().numpy()
            pred_bin_img = pred_bin[:,0,:,:].permute(1,2,0).squeeze(2).cpu().numpy()
            plt.imshow(tgt_img)
            plt.axis('off')
            plt.savefig(f"{save_file}tgt.png")
            plt.close()
            plt.imshow(pred_img)
            plt.axis('off')
            plt.savefig(f"{save_file}pred.png")
            plt.close()
            plt.imshow(pred_bin_img)
            plt.axis('off')
            plt.savefig(f"{save_file}bin.png")
            plt.close()

# log and print info simultaneously
def log(msg, logger, print_msg=True, log_msg=True):
    if print_msg:
        print(msg)
    if log_msg:
        logger.error(msg)
