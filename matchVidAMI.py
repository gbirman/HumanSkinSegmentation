
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle 

# get path to training/testing image/annotation given an index number
def getImgPath(num, dataType, imgType, root="/Users/gabrielbirman/COS429/SkinSegmentation/Data/VDM"):

    # formats the index number into the proper name for a train file 
    def getTrainFile(num):
        return f"AMI ({num}).png"

    # formats the index number into the proper name for a test file 
    def getTestFile(num):
        return f"{num}0000.png"

    # ensure img type flags are set correctly
    if imgType != 'raw' and imgType != 'ann':
        raise ValueError

    # get corresponding file name 
    if dataType == 'train':
        fname = getTrainFile(num)
    elif dataType == 'test':
        fname = getTestFile(num)
    else:
        raise ValueError

    p = Path(root)
    path = p / dataType / f"{dataType}_AMI" / imgType / fname
    return path

# get path to video given an index number and letter
def getVidPath(num, letter, root="/Users/gabrielbirman/COS429/SkinSegmentation/Data/AMI"):

    # formats the index number into the proper name (no file extension)
    def getVidName(num, letter):
        return f"IS100{num}{letter}"

    # formats the index number into the proper name
    def getVidFile(num, letter):
        return f"{getVidName(num, letter)}.C.avi"
    
    p = Path(root)
    path = p / getVidName(num, letter) / 'video' / getVidFile(num, letter)
    return path

img_folder = "/Users/gabrielbirman/COS429/SkinSegmentation/Data/VDM"
vid_folder = "/Users/gabrielbirman/COS429/SkinSegmentation/Data/AMI"

# manually observed data matching annotated images to the videos they were captured from 
img_idx_to_vid_idx = {
                        1 : (0, 'a'), 2: (0, 'a'), 3: (1, 'a'), 4: (1, 'a'), 5: (2, 'b'), 6: (2, 'b'), 
                        7 : (3, 'a'), 8: (3, 'a'), 9: (4, 'a'), 10: (4, 'a'), 11: (5, 'a'), 12: (5, 'a'), 
                        13 : (6, 'a'), 14: (6, 'a'), 15: (7, 'a'), 16: (7, 'a'), 17: (8, 'a'), 18: (8, 'a'), 
                        19 : (9, 'a'), 20: (9, 'a'), 21: (0, 'a'), 22: (2, 'b'), 23: (4, 'a'), 24: (6, 'a'), 25: (8, 'a')
                    }

### Save mappings from annotated image index to logical mask ###

def saveMasks(dict_save_folder=Path("/Users/gabrielbirman/COS429/SkinSegmentation/Matching/AMI")):
    img_idx_to_mask_train = {}
    img_idx_to_mask_test = {}

    for img_idx in img_idx_to_vid_idx.keys():

        # get train/test annotations 
        train_ann_path = getImgPath(img_idx, 'train', 'ann', img_folder)
        test_ann_path = getImgPath(img_idx, 'test', 'ann', img_folder)

        train_ann = cv2.imread(str(train_ann_path))
        test_ann = cv2.imread(str(test_ann_path))

        # convert annotated images to logical arrays
        # NOTE: # cv2 loads in BGR image 
        train_mask = np.all(train_ann == [0, 0, 255], axis=2) 
        test_mask = np.all(test_ann == [0, 0, 255], axis=2) 

        # save key-value pair to dictionary
        img_idx_to_mask_train[img_idx] = train_mask
        img_idx_to_mask_test[img_idx] = test_mask

    # save dictionaries 
    with open(dict_save_folder / 'img_idx_to_mask_train.pickle', 'wb') as handle:
        pickle.dump(img_idx_to_mask_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dict_save_folder / 'img_idx_to_mask_test.pickle', 'wb') as handle:
        pickle.dump(img_idx_to_mask_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

### Save mappings from extracted image index to corresponding frame in original video ###

def saveFrames(dict_save_folder=Path("/Users/gabrielbirman/COS429/SkinSegmentation/Matching/AMI"), 
                frame_save_folder=Path("/Users/gabrielbirman/COS429/SkinSegmentation/Matching/AMI/FrameMatch")):
    img_idx_to_frame_idx_train = {}
    img_idx_to_frame_idx_test = {}

    for img_idx, vid_idx in img_idx_to_vid_idx.items():

        print(f"image index: {img_idx}")

        # get raw train/test images 
        train_raw_path = getImgPath(img_idx, 'train', 'raw', img_folder)
        test_raw_path = getImgPath(img_idx, 'test', 'raw', img_folder)

        train_raw = cv2.imread(str(train_raw_path))
        test_raw = cv2.imread(str(test_raw_path))

        # get video 
        vid_path = getVidPath(*vid_idx, vid_folder)
        vid = cv2.VideoCapture(str(vid_path))

        startFrame = 0 # which frame to start the video from 
        endFrame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 # non-inclusive last frame of the video (default is number of frames in the video)
        vid.set(1, startFrame) # cv2.CAP_PROP_FRAME_COUNT not working 

        # find frame in video that best matches raw images 
        min_loss_train = np.inf
        min_frame_train = 0
        min_loss_test = np.inf
        min_frame_test = 0
        train_frame = None 
        test_frame = None 
        for frame_idx in range(startFrame, endFrame): # iterate through video frames 

            ret, frame = vid.read()
            if ret is False: 
                print(f"frame index {frame_idx} outside of video range with end frame {endFrame} at image index {img_idx}")
                break

            # minimize training loss 
            train_loss = np.sum((frame - train_raw)**2) / np.sqrt(np.sum(frame**2) * np.sum(train_raw**2))
            if train_loss < min_loss_train:
                min_loss_train = train_loss 
                min_frame_train = frame_idx
                train_frame = frame 
            
            # minimize testing loss 
            test_loss = np.sum((frame - test_raw)**2) / np.sqrt(np.sum(frame**2) * np.sum(test_raw**2))
            if test_loss < min_loss_test:
                min_loss_test = test_loss 
                min_frame_test = frame_idx    
                test_frame = frame     

        # # save corresponding images 
        # # NOTE: this is for manually inspecting image similarity 
        # cv2.imwrite(frame_save_folder / f"train_{img_idx}.png", train_frame)
        # cv2.imwrite(frame_save_folder / f"test_{img_idx}.png", test_frame)

        # save key-value pair to dictionary 
        img_idx_to_frame_idx_train[img_idx] = min_frame_train
        img_idx_to_frame_idx_test[img_idx] = min_frame_test

    # save dictionaries   
    with open(dict_save_folder / 'img_idx_to_frame_idx_train.pickle', 'wb') as handle:
        pickle.dump(img_idx_to_frame_idx_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dict_save_folder / 'img_idx_to_frame_idx_test.pickle', 'wb') as handle:
        pickle.dump(img_idx_to_frame_idx_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    saveFrames()