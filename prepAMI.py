
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle 
from matchVidAMI import getImgPath, getVidPath, img_idx_to_vid_idx, img_folder, vid_folder
import logging

# log and print info simultaneously
def log(msg):
    print(msg)
    logger.error(msg)

# load in dictionaries 
load_folder = Path("/Users/gabrielbirman/COS429/SkinSegmentation/Matching/AMI")
img_idx_to_frame_idx_train = np.load(load_folder / 'img_idx_to_frame_idx_train.pickle', allow_pickle=True)
img_idx_to_frame_idx_test = np.load(load_folder / 'img_idx_to_frame_idx_test.pickle', allow_pickle=True)

min_step, max_step = 25 // 4, 25 * 5
step_num = 3
num_inp = 10
save_folder = Path("/Users/gabrielbirman/COS429/SkinSegmentation/Data/VidSequences/AMI")
inp_list, gt_list = [], []
train_idx, test_idx = [], []
delta = []

logging.basicConfig(filename=f"/Users/gabrielbirman/COS429/SkinSegmentation/Data/VidSequences/AMI/{step_num}.txt", filemode='w', format="%(asctime)s;%(message)s", level=logging.ERROR)
logger = logging.getLogger()

# training
for img_idx, vid_idx in img_idx_to_vid_idx.items():

    # get video 
    vid_path = getVidPath(*vid_idx, vid_folder)
    vid = cv2.VideoCapture(str(vid_path))

    flag = True
    num_attempts = 0
    num_good = 0
    while flag:
        try: 
            step_size = np.random.randint(min_step, max_step)

            # train
            time_series = [] 

            startFrame = img_idx_to_frame_idx_train[img_idx]

            # frames before center 
            for i in range(1, step_num//2 + 1).__reversed__():
                vid.set(1, startFrame - i * step_size)
                _, frame = vid.read(startFrame)
                _, buf = cv2.imencode(".png", frame) 
                time_series.append(buf)

            # center frame 
            vid.set(1, startFrame)
            _, frame = vid.read()
            _, buf = cv2.imencode(".png", frame) 
            time_series.append(buf)
            
            # frames after center
            for i in range(1, step_num//2 + 1):
                vid.set(1, startFrame + i * step_size)
                _, frame = vid.read(startFrame)
                _, buf = cv2.imencode(".png", frame) 
                time_series.append(buf)

            inp_list.append(time_series)

            # groundtruth 
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
for img_idx, vid_idx in img_idx_to_vid_idx.items():

    # get video 
    vid_path = getVidPath(*vid_idx, vid_folder)
    vid = cv2.VideoCapture(str(vid_path))

    flag = True
    num_attempts = 0
    num_good = 0
    while flag:
        try: 
            step_size = np.random.randint(min_step, max_step)

            # train
            time_series = [] 

            startFrame = img_idx_to_frame_idx_test[img_idx]

            # frames before center 
            for i in range(1, step_num//2 + 1).__reversed__():
                vid.set(1, startFrame - i * step_size)
                _, frame = vid.read(startFrame)
                _, buf = cv2.imencode(".png", frame) 
                time_series.append(buf)

            # center frame 
            vid.set(1, startFrame)
            _, frame = vid.read()
            _, buf = cv2.imencode(".png", frame) 
            time_series.append(buf)
            
            # frames after center
            for i in range(1, step_num//2 + 1):
                vid.set(1, startFrame + i * step_size)
                _, frame = vid.read(startFrame)
                _, buf = cv2.imencode(".png", frame) 
                time_series.append(buf)

            inp_list.append(time_series)

            # groundtruth 
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

data_file = save_folder / f"AMI_Data_{step_num}.npz"
np.savez_compressed(data_file, inp=inp_list, gt=gt_list) 
np.savez(save_folder / f"AMI_Data_{step_num}_indices.npz", train=train_idx, test=test_idx)
np.savez(save_folder / f"AMI_Data_{step_num}_delta.npz", delta=delta)
    
if __name__ == "__main__":
    pass