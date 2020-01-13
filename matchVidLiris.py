
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle 

# get path to training/testing image/annotation given an index number
def getImgPath(num, dataType, imgType, root="/Users/gabrielbirman/COS429/SkinSegmentation/Data/VDM"):

    # formats the index number into the proper name for a train file 
    def getTrainFile(num):
        return f"img{num}.png"

    # formats the index number into the proper name for a test file 
    def getTestFile(num):
        return f"imgLIRIS{num:02d}.png"

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
    path = p / dataType / f"{dataType}_LIRIS" / imgType / fname
    return path

img_folder = "/Users/gabrielbirman/COS429/SkinSegmentation/Data/VDM"
vid_folder = "/Users/gabrielbirman/COS429/SkinSegmentation/Data/LIRIS/D2"

### Save mappings from annotated image index to logical mask ###

def saveMasks(dict_save_folder = Path("/Users/gabrielbirman/COS429/SkinSegmentation/Matching/LIRIS")):
    img_idx_to_mask_train = {}
    img_idx_to_mask_test = {}

    # training images 
    for img_idx in range(1,31): #[1,30]

        # get train/test annotations 
        train_ann_path = getImgPath(img_idx, 'train', 'ann', img_folder)

        train_ann = cv2.imread(str(train_ann_path))

        # convert annotated images to logical arrays
        # NOTE: # cv2 loads in BGR image 
        train_mask = np.all(train_ann == [0, 0, 255], axis=2) 

        # save key-value pair to dictionary
        img_idx_to_mask_train[img_idx] = train_mask

    # testing images 
    for img_idx in range(1,26): #[1,25]

        # get train/test annotations 
        test_ann_path = getImgPath(img_idx, 'test', 'ann', img_folder)

        test_ann = cv2.imread(str(test_ann_path))

        # convert annotated images to logical arrays
        # NOTE: # cv2 loads in BGR image 
        test_mask = np.all(test_ann == [0, 0, 255], axis=2) 

        # save key-value pair to dictionary
        img_idx_to_mask_test[img_idx] = test_mask

    # save dictionaries 
    with open(dict_save_folder / 'img_idx_to_mask_train.pickle', 'wb') as handle:
        pickle.dump(img_idx_to_mask_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dict_save_folder / 'img_idx_to_mask_test.pickle', 'wb') as handle:
        pickle.dump(img_idx_to_mask_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

### Save mappings from extracted image index to corresponding frame in original video ###

def saveFrames(dict_save_folder = Path("/Users/gabrielbirman/COS429/SkinSegmentation/Matching/LIRIS"), 
                frame_save_folder = Path("/Users/gabrielbirman/COS429/SkinSegmentation/Matching/LIRIS/FrameMatch")):

   
    # intialize data  
    train_data = {img_idx:{'loss': np.inf, 'best': None} for img_idx in range(1,31)}
    test_data = {img_idx:{'loss': np.inf, 'best': None} for img_idx in range(1,26)}

    # iterate through all frames and minimize MSE loss
    for frame_name in Path(vid_folder).glob("**/*.jpg"): 

        frame = cv2.imread(str(frame_name))

        # training images 
        for img_idx in range(1,31):
            img_path = getImgPath(img_idx, 'train', 'raw', img_folder)
            img = cv2.imread(str(img_path))
            loss = np.sum((frame - img)**2) / np.sqrt(np.sum(frame**2) * np.sum(img**2))
            if loss < train_data[img_idx]['loss']:
                train_data[img_idx]['loss'] = loss 
                train_data[img_idx]['best'] = str(frame_name)

        # testing images 
        for img_idx in range(1,26):
            img_path = getImgPath(img_idx, 'test', 'raw', img_folder)
            img = cv2.imread(str(img_path))
            loss = np.sum((frame - img)**2) / np.sqrt(np.sum(frame**2) * np.sum(img**2))
            if loss < test_data[img_idx]['loss']:
                test_data[img_idx]['loss'] = loss 
                test_data[img_idx]['best'] = str(frame_name)

    # prepare dictionaries
    img_idx_to_frame_train = {img_idx:train_data[img_idx]['best'] for img_idx in range(1,31)}
    img_idx_to_frame_test = {img_idx:test_data[img_idx]['best'] for img_idx in range(1,26)}

    # save corresponding images for testing 
    for img_idx in range(1, 31):
        frame = cv2.imread(train_data[img_idx]['best'])
        cv2.imwrite(str(frame_save_folder / f"train_{img_idx}.jpg"), frame)

    for img_idx in range(1, 26):
        frame = cv2.imread(test_data[img_idx]['best'])
        cv2.imwrite(str(frame_save_folder / f"test_{img_idx}.jpg"), frame)

    # save dictionaries   
    with open(dict_save_folder / 'img_idx_to_frame_train.pickle', 'wb') as handle:
        pickle.dump(img_idx_to_frame_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dict_save_folder / 'img_idx_to_frame_test.pickle', 'wb') as handle:
        pickle.dump(img_idx_to_frame_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    saveFrames()