
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle 
from matchVidLiris import getImgPath, img_folder
import parse
import logging

# log and print info simultaneously
def log(msg):
    print(msg)
    logger.error(msg)

# get path to video given an index number and letter
def getVidPath(vid_idx, frame_idx, root="/Users/gabrielbirman/COS429/SkinSegmentation/Data/LIRIS/D2"):

    # formats the index number into the proper file
    def getVidFile(vid_idx, frame_idx):
        return f"vid{vid_idx:04d}/rgb-{frame_idx:06d}.jpg"

    p = Path(root)
    path = p / getVidFile(vid_idx, frame_idx)
    return path

# parse the string file to get the vid/frame indices 
def getVidIdx(input_string, root="/Users/gabrielbirman/COS429/SkinSegmentation/Data/LIRIS/D2"):
    format_string = str(Path(root) / "vid{:04d}/rgb-{:06d}.jpg")
    parsed = parse.parse(format_string, input_string)
    vid_idx, frame_idx = parsed[0], parsed[1]
    return vid_idx, frame_idx

# load in dictionaries 
load_folder = Path("/Users/gabrielbirman/COS429/SkinSegmentation/Matching/LIRIS")
img_idx_to_frame_train = np.load(load_folder / 'img_idx_to_frame_train.pickle', allow_pickle=True)
img_idx_to_frame_test = np.load(load_folder / 'img_idx_to_frame_test.pickle', allow_pickle=True)

min_step, max_step = 25 // 4, 25 * 5
step_num = 3
num_inp = 10
save_folder = Path("/Users/gabrielbirman/COS429/SkinSegmentation/Data/VidSequences/LIRIS")
inp_list, gt_list = [], []
train_idx, test_idx = [], []
delta = [] 

logging.basicConfig(filename=f"/Users/gabrielbirman/COS429/SkinSegmentation/Data/VidSequences/LIRIS/{step_num}.txt", filemode='w', format="%(asctime)s;%(message)s", level=logging.ERROR)
logger = logging.getLogger()

# training
for img_idx, frame in img_idx_to_frame_train.items():

    vid_idx, startFrame = getVidIdx(frame)
    flag = True
    num_attempts = 0
    num_good = 0
    while flag:
        try: 
            step_size = np.random.randint(min_step, max_step)

            time_series = []

            for i in range(1, step_num//2 + 1).__reversed__():
                name = getVidPath(vid_idx, startFrame - i * step_size)
                buf = open(name, 'rb').read()
                time_series.append(buf)

            name = getVidPath(vid_idx, startFrame)
            buf = open(name, 'rb').read()
            time_series.append(buf)

            for i in range(1, step_num//2 + 1):
                name = getVidPath(vid_idx, startFrame + i * step_size)
                buf = open(name, 'rb').read()
                time_series.append(buf)

            inp_list.append(time_series)

            img_path = getImgPath(img_idx, 'train', 'ann', img_folder)
            img = cv2.imread(str(img_path))
            img = np.all(img == [0, 0, 255], axis=2)
            gt_list.append(img)

            log(f"Train index: {img_idx}, step size = {step_size}")
            train_idx.append(len(inp_list)-1)
            delta.append(step_size)

            num_good += 1 
            if num_good >= num_inp:
                flag = False
        except:
            num_attempts += 1
            if num_attempts >= 100:
                log(f"Invalid train index: {img_idx}, Start Frame = {startFrame}")
                flag = False

# testing
for img_idx, frame in img_idx_to_frame_test.items():

    vid_idx, startFrame = getVidIdx(frame)
    flag = True
    num_attempts = 0
    num_good = 0
    while flag:
        try: 
            step_size = np.random.randint(min_step, max_step)

            time_series = []

            for i in range(1, step_num//2 + 1).__reversed__():
                name = getVidPath(vid_idx, startFrame - i * step_size)
                buf = open(name, 'rb').read()
                time_series.append(buf)

            name = getVidPath(vid_idx, startFrame)
            buf = open(name, 'rb').read()
            time_series.append(buf)

            for i in range(1, step_num//2 + 1):
                name = getVidPath(vid_idx, startFrame + i * step_size)
                buf = open(name, 'rb').read()
                time_series.append(buf)

            inp_list.append(time_series)

            img_path = getImgPath(img_idx, 'test', 'ann', img_folder)
            img = cv2.imread(str(img_path))
            img = np.all(img == [0, 0, 255], axis=2)
            gt_list.append(img)

            log(f"Test index: {img_idx}, step size = {step_size}")
            test_idx.append(len(inp_list)-1)
            delta.append(step_size)

            num_good += 1 
            if num_good >= num_inp:
                flag = False
        except:
            num_attempts += 1
            if num_attempts >= 100:
                log(f"Invalid test index: {img_idx}, Start Frame = {startFrame}")
                flag = False

data_file = save_folder / f"LIRIS_Data_{step_num}.npz"
np.savez_compressed(data_file, inp=inp_list, gt=gt_list) 
np.savez(save_folder / f"LIRIS_Data_{step_num}_indices.npz", train=train_idx, test=test_idx)   
np.savez(save_folder / f"LIRIS_Data_{step_num}_delta.npz", delta=delta)

if __name__ == "__main__":
    pass