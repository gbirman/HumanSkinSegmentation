from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt 
import pandas 
from abc import ABC, abstractmethod
import pickle


# use this method to instantiate a dataset 
def get_dataset(dataset_name, **kwargs):
    if dataset_name == "FSD":
        dataset = FSD_Dataset(**kwargs)
    elif dataset_name == "Pratheepan_Total":
        dataset = Pratheepan_Total_Dataset(**kwargs)
    elif dataset_name == "Pratheepan_Face":
        dataset = Pratheepan_Face_Dataset(**kwargs)
    elif dataset_name == "Pratheepan_Family":
        dataset = Pratheepan_Family_Dataset(**kwargs)
    elif dataset_name == "AMI":
        dataset = AMI_dataset(**kwargs)
    elif dataset_name == "LIRIS":
        dataset = LIRIS_dataset(**kwargs) 
    else:
        raise ValueError("Invalid dataset name.")
    return dataset

# container class for image datasets
class ImageDataset(ABC, Dataset):

    def __init__(self, output_size=50, stride=10, mode="train", 
        data_root="./Data", createDataFile=False, grayscale=False, use_sigmoid=False, just_subtract=True,
        original=False, **kwargs): 

        # ensure necessary class attributes are set 
        try:
            self._folder_name
            self._file_name
            self._inp_folder_names
            self._gt_folder_names 
            self._mean 
            self._std 
        except:
            raise AttributeError("Ensure all necessary class attributes are set.")

        print("Initializing Dataset ...")
        super().__init__()
        # superclass attributes 
        self.output_size = output_size # size of smaller side of image
        self.stride = stride # stride when collecting output_size x output_size images
        self.mode = mode # multi-stride inputs when true 
        # subclass attributes 
        dataset_folder = Path(data_root) / self._folder_name
        self.input_folders = [dataset_folder / fn for fn in self._inp_folder_names] 
        self.groundtruth_folders = [dataset_folder / fn for fn in self._gt_folder_names]
        self.data_file = dataset_folder / self._file_name  
        self.createDataFile = createDataFile
        self.grayscale = grayscale
        self.use_sigmoid = use_sigmoid
        self.just_subtract = just_subtract
        self.original = original
        self.inp, self.gt = self._load_data()
        
        print("Initialization Complete!")

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.inp) 

    # saves images buffers to data file  
    def _save_data(self):
        # takes a minute or two to save (locally) but it could save on I/O costs when 
        # constantly reading in images from remote dir

        data_file, input_folders, groundtruth_folders = self.data_file, self.input_folders, self.groundtruth_folders

        inp_list, gt_list = [], []
        for idx, input_folder in enumerate(input_folders):
            groundtruth_folder = groundtruth_folders[idx]
            input_iter = self._get_input_iter(input_folder)
            for input_path in input_iter:
                groundtruth_path = groundtruth_folder / self._label_equiv(input_path.stem)

                inp_buf = open(input_path, 'rb').read()
                gt_buf = open(groundtruth_path, 'rb').read()

                inp_list.append(inp_buf)
                gt_list.append(gt_buf)

        np.savez_compressed(data_file, inp=inp_list, gt=gt_list) 
        print(f"Dataset compressed and saved to {data_file}")

        return inp_list, gt_list 

    # return the pandas dataframe containing raw/groundtruth files paths 
    def _load_data(self):

        # unpack dataset attributes
        data_file, createDataFile = self.data_file, self.createDataFile

        # create dataframe 
        if not Path(data_file).exists() or createDataFile:
            return self._save_data()

        # load in existing datafile (will take some to load into RAM)
        loaded = np.load(data_file)
        return loaded['inp'], loaded['gt']
    
    # return input/label numpy image pair at index 
    def _get_images(self, index):

        # unpack dataset attributes
        inp, gt, grayscale = self.inp, self.gt, self.grayscale

        input_img = cv2.imdecode(np.frombuffer(inp[index], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        num_channels = input_img.shape[2]
        if grayscale and num_channels == 3:
            # scale [0,255] to [0.0, 1.0]
            input_img = input_img / 255.0
            self._mean = np.array(self._mean) / 255.0
            self._std = np.array(self._std) / 255.0
            B, G, R = input_img[:,:,0], input_img[:,:,1], input_img[:,:,2]
            input_img = 0.299*R + 0.587*G + 0.114*B
        groundtruth_img = cv2.imdecode(np.frombuffer(gt[index], dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        return input_img, groundtruth_img

    # display input/groundtruth pair for tensors with dim = 3 (C x H x W)
    # or display multiple input/groundtruth pairs for tensors with dim = 4 (N x C x H x W)
    def _disp_imgs(self, input_tensor, groundtruth_tensor):
        print(input_tensor.shape, groundtruth_tensor.shape)
        if input_tensor.dim() == 3: 
            inp, gt = self._tensor_to_img(input_tensor, groundtruth_tensor)
            print(inp.shape, gt.shape)
            _, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(inp)
            ax2.imshow(gt)
            plt.show()
        elif input_tensor.dim() == 4: 
            for i in range(input_tensor.shape[0]): 
                self._disp_imgs(input_tensor[i,...], groundtruth_tensor[i,...])
        elif input_tensor.dim() == 5:
            for i in range(input_tensor.shape[0]): 
                _, axs = plt.subplots(input_tensor.shape[2], 2)
                for j in range(input_tensor.shape[2]):
                    inp, gt = self._tensor_to_img(input_tensor[i,:,j,:,:], groundtruth_tensor[i,...])
                    axs[j][0].imshow(inp)
                    axs[j][1].imshow(gt)
                plt.savefig(f"{i}.png")
        else:
            raise ValueError("Tensor have 3 or 4 dims.")

    # resize the images input_img, groundtruth_img
    def _resize_images(self, input_img, groundtruth_img):
    
        # unpack dataset attributes 
        grayscale, output_size = self.grayscale, self.output_size
        
        if isinstance(input_img, list):
            scale_factor = output_size / np.min([input_img[0].shape[0], input_img[0].shape[1]])
            input_img = [cv2.resize(im, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR) for im in input_img]
            if self.original:
                groundtruth_img = cv2.resize(groundtruth_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR) 
            else:
                groundtruth_img = cv2.resize(groundtruth_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST) 
            return input_img, groundtruth_img

        # resize images such that smaller dimension is of size output_size
        scale_factor = output_size / np.min([input_img.shape[0], input_img.shape[1]])
        input_img = cv2.resize(input_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        if self.original:
            groundtruth_img = cv2.resize(groundtruth_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR) 
        else:
            groundtruth_img = cv2.resize(groundtruth_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

        # add single channel to grayscale image
        if grayscale:
            input_img = np.expand_dims(input_img, -1)

        return input_img, groundtruth_img

    # normalize input tensor of shape C x H x W
    def _normalize(self, input_tensor):
        grayscale, mean, std, just_subtract = self.grayscale, self._mean, self._std, self.just_subtract
        
        if input_tensor.dim() == 4: # C x D x H x W
            mean = torch.as_tensor(mean, dtype=torch.float32, device=input_tensor.device)
            std = torch.as_tensor(std, dtype=torch.float32, device=input_tensor.device)
            input_tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
            return

        num_channels = len(mean)
        if just_subtract:  
            # only subtract the average intensity from the entire image 
            if grayscale:
                if num_channels == 3:
                    mean = [np.mean(mean)]
                std = [1.]
            else:
                mean = [np.mean(mean)] * 3
                std = [1.] * 3 
        elif grayscale and num_channels == 3:
            # we can use linear scaling of mean/std property
            BGR_to_Y = [0.114, 0.587, 0.299]
            mean = [np.dot(BGR_to_Y, mean)]
            std = [np.dot(BGR_to_Y, std)]

        torchvision.transforms.functional.normalize(input_tensor, mean, std, inplace=True)

    # stride inputs along larger dimension to get multiple smaller images 
    def _get_multi(self, input_tensor, groundtruth_tensor):
        
        # unpack dataset attributes 
        output_size, stride = self.output_size, self.stride

        # get smaller dimension between width/height and scaling factor for resizing
        if input_tensor.dim() == 4:
            H_dim, W_dim = 2, 3 
        elif input_tensor.dim() == 5:
            H_dim, W_dim = 3, 4
        larger_dim = H_dim if input_tensor.shape[H_dim] >= input_tensor.shape[W_dim] else W_dim

        # number of strides to take 
        num_strides = int( np.floor( (input_tensor.shape[larger_dim] - (output_size-1) - 1)/stride + 1 ) )

        # prepare multi-stride tensor  
        if input_tensor.dim() == 4:
            input_tensor_multi = input_tensor.new_empty((num_strides, input_tensor.shape[1], output_size, output_size))
            groundtruth_tensor_multi = groundtruth_tensor.new_empty((num_strides, groundtruth_tensor.shape[1], output_size, output_size))
        elif input_tensor.dim() == 5:   
            input_tensor_multi = input_tensor.new_empty((num_strides, input_tensor.shape[1], input_tensor.shape[2], output_size, output_size))
            groundtruth_tensor_multi = groundtruth_tensor.new_empty((num_strides, groundtruth_tensor.shape[1], groundtruth_tensor.shape[2], output_size, output_size))
        for i in range(num_strides):
            start_idx = stride * i 
            indices = torch.arange(start_idx, start_idx+output_size)
            input_tensor_multi[i,...] = torch.index_select(input_tensor, larger_dim, indices)
            groundtruth_tensor_multi[i,...] = torch.index_select(groundtruth_tensor, larger_dim, indices)

        # display images
        # self._disp_imgs(input_tensor_multi, groundtruth_tensor_multi)

        return input_tensor_multi, groundtruth_tensor_multi  

    # process tensors for data loaders 
    def _prepare(self, input_tensor, groundtruth_tensor):

        # unpack dataset attributes 
        mode = self.mode

        input_tensor = input_tensor.float() # C x H x W (float32)
        groundtruth_tensor = groundtruth_tensor.float() 
        groundtruth_tensor /= torch.max(groundtruth_tensor)

        # normalize input X ~ N(0,1) across all three BGR color channels 
        # NOTE: OpenCV converts to grayscale using the following formula: Y = 0.299 R + 0.587 G + 0.114B
        # NOTE: make sure this works 
        self._normalize(input_tensor)

        # Add dimension N: N x C x H x W 
        input_tensor.unsqueeze_(0) 
        groundtruth_tensor.unsqueeze_(0)
        
        # get multiple training images by striding image along larger dimension
        if mode == "train":
            input_tensor, groundtruth_tensor = self._get_multi(input_tensor, groundtruth_tensor)

        return input_tensor, groundtruth_tensor

    # return the input/label tensor pair at index 
    def __getitem__(self, index):
        input_img, groundtruth_img = self._get_images(index)
        if self.mode == "test":
            input_img, _ = self._resize_images(input_img, groundtruth_img)
        else:
            input_img, groundtruth_img = self._resize_images(input_img, groundtruth_img)
        input_tensor, groundtruth_tensor = self._img_to_tensor(input_img, groundtruth_img)
        input_tensor, groundtruth_tensor = self._prepare(input_tensor, groundtruth_tensor)
        return input_tensor, groundtruth_tensor

    # return an iterator over images in the input folder
    @abstractmethod
    def _get_input_iter(self, input_folder):
        pass

    # given an input image stem, return its corresponding label image
    @abstractmethod
    def _label_equiv(self, img_stem):
        pass
    
    # convert input images to tensors in C x H x W format 
    # inputs/label parameters should be uint8
    # inputs/label outputs should be uint8, boolean respectively
    # Skin pixels set to True 
    @abstractmethod 
    def _img_to_tensor(self, input_img, groundtruth_img):
        pass 

    # inverse of _img_to_tensor
    @abstractmethod 
    def _tensor_to_img(self, input_tensor, groundtruth_tensor):
        pass 

# FSD dataset all imgs
class FSD_Dataset(ImageDataset):

    # Dataset info
    _folder_name = "FSD" # name of folder containing dataset in data_root (which contains folder with ALL datasets)
    _file_name = "FSD_Data.npz" # name of file containing all data, label pairs in dataset 
    _inp_folder_names = ["Original"] 
    _gt_folder_names = ["Skin"] 

    # Dataset Statistics (BGR order) 
    _mean = [102.9, 111.5, 126.3]  
    _std = [72.2, 70.6, 74.8]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # inputs in the form:
    # input_img: BGR, H x W x C = ? x ? x 3, [0, 255] unit8
    # groundtruth_img: Binary, H x W = ? x ?, [0, 255] uint8
    def _img_to_tensor(self, input_img, groundtruth_img):
        # outputs in the form: 
        # input_tensor: BGR, C x H x W  = 3 x ? x ?, [0, 255] uint8 
        # groundtruth_tensor: Binary, C x H x W = 1 x ? x ?, [0, 255] uint8 
        input_tensor = torch.from_numpy(input_img).permute(2,0,1)
        groundtruth_tensor = ~torch.from_numpy(groundtruth_img).unsqueeze(0).byte()
        return input_tensor, groundtruth_tensor

    # inputs in the form: 
    # input_tensor: BGR, C x H x W  = 3 x ? x ?, [0, 255] uint8 
    # groundtruth_tensor: Binary, C x H x W = 1 x ? x ?, [0, 255] uint8 
    def _tensor_to_img(self, input_tensor, groundtruth_tensor):
        # outputs in the form:
        # input_img: BGR, H x W x C = ? x ? x 3, [0, 255] unit8
        # groundtruth_img: Binary, H x W = ? x ?, [0, 255] uint8
        input_img = input_tensor.permute(1,2,0).numpy()
        groundtruth_img = groundtruth_tensor.squeeze(0).byte().numpy()
        return input_img, groundtruth_img

    # return an iterator over images in the input folder
    def _get_input_iter(self, input_folder):
            return input_folder.glob("*")

    # given an input image stem, return its corresponding label image
    def _label_equiv(self, img_stem):
        return f"{img_stem}_s.png"

# Intermediate class for Pratheepan Datset
class Pratheepan_Dataset(ImageDataset):

    # Dataset info
    _folder_name = "Pratheepan" # name of folder containing dataset in data_root (which contains folder with ALL datasets)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # inputs in the form:
    # input_img: BGR, H x W x C = ? x ? x 3, [0, 255] unit8
    # groundtruth_img: BGRA, H x W x C = ? x ? x 3, [0, 255] unit8
    def _img_to_tensor(self, input_img, groundtruth_img):
        # outputs in the form: 
        # input_tensor: BGR, C x H x W  = 3 x ? x ?, [0, 255] uint8 
        # groundtruth_tensor: Binary, C x H x W = 1 x ? x ?, [0, 255] uint8 
        input_tensor = torch.from_numpy(input_img).permute(2,0,1)
        groundtruth_tensor = torch.from_numpy(groundtruth_img[:,:,0]).unsqueeze(0).byte()
        return input_tensor, groundtruth_tensor

    # inputs in the form: 
    # input_tensor: BGR, H x W x C = ? x ? x 3, [0, 255] unit8
    # groundtruth_tensor: Binary, C x H x W = 1 x ? x ?, [0, 255] uint8 
    def _tensor_to_img(self, input_tensor, groundtruth_tensor):
        # outputs in the form:
        # input_img: BGR, H x W x C = ? x ? x 3, [0, 255] unit8
        # groundtruth_img: Binary, H x W = ? x ?, [0, 255] uint8
        input_img = input_tensor.permute(1,2,0).numpy()
        groundtruth_img = groundtruth_tensor.squeeze(0).byte().numpy()
        return input_img, groundtruth_img

    # return an iterator over images in the input folder
    def _get_input_iter(self, input_folder):
            return input_folder.glob("*")

    # given an input image stem, return its corresponding label image
    def _label_equiv(self, img_stem):
        return f"{img_stem}.png"

# Pratheepan dataset all imgs
class Pratheepan_Total_Dataset(Pratheepan_Dataset):

    # Dataset info
    _file_name = "Pratheepan_Total_Data.npz" # name of file containing all data, label pairs in dataset 
    _inp_folder_names = ["Pratheepan_Dataset/FacePhoto", "Pratheepan_Dataset/FamilyPhoto"] 
    _gt_folder_names = ["Ground_Truth/GroundT_FacePhoto","Ground_Truth/GroundT_FamilyPhoto"] 

    # Dataset Statistics (BGR order) 
    # this ensures that we normalize according to the training dataset stats (FSD)
    _mean = FSD_Dataset._mean #[99.1, 110.2, 132.6] 
    _std = FSD_Dataset._std #[73.4, 73.1, 78.2]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# Pratheepan dataset face imgs 
class Pratheepan_Face_Dataset(Pratheepan_Dataset):

    # Dataset info
    _file_name = "Pratheepan_Face_Data.npz" # name of file containing all data, label pairs in dataset 
    _inp_folder_names = ["Pratheepan_Dataset/FacePhoto"] 
    _gt_folder_names = ["Ground_Truth/GroundT_FacePhoto"] 

    # Dataset Statistics (BGR order) 
    _mean = FSD_Dataset._mean #[102.6, 117.7, 145.8] 
    _std = FSD_Dataset._std #[77.5, 77.2, 82.3]

    def __init__(self, **kwargs):
        super().__init__(**kwargs) 

# Pratheepan dataset family imgs
class Pratheepan_Family_Dataset(Pratheepan_Dataset):

    # Dataset info
    _file_name = "Pratheepan_Family_Data.npz" # name of file containing all data, label pairs in dataset 
    _inp_folder_names = ["Pratheepan_Dataset/FamilyPhoto"] 
    _gt_folder_names = ["Ground_Truth/GroundT_FamilyPhoto"] 

    # Dataset Statistics (BGR order) 
    _mean = FSD_Dataset._mean #[96.7, 104.9, 123.4]
    _std = FSD_Dataset._std #[70.3, 69.6, 73.9]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)   

class AMI_dataset(ImageDataset):

    # Dataset info
    _folder_name = "VidSequences/AMI" # name of folder containing dataset in data_root (which contains folder with ALL datasets)
    _inp_folder_names = [] 
    _gt_folder_names = [] 

    # Dataset Statistics (BGR order) 
    _mean = FSD_Dataset._mean
    _std = FSD_Dataset._std

    def __init__(self, frame_num=5, **kwargs):
        self._file_name = f"AMI_Data_{frame_num}.npz"
        self.frame_num = frame_num
        super().__init__(**kwargs) 

    # return the pandas dataframe containing raw/groundtruth files paths 
    def _load_data(self):

        # unpack dataset attributes
        data_file = self.data_file

        # create dataframe 
        if not Path(data_file).exists():
            print("Provide the data file for the AMI dataset")
            raise ValueError

        # load in existing datafile (will take some to load into RAM)
        loaded = np.load(data_file, allow_pickle=True)
        return loaded['inp'], loaded['gt']

    # return input/label numpy image pair at index 
    def _get_images(self, index):

        # unpack dataset attributes
        inp, gt = self.inp, self.gt

        input_img = [cv2.imdecode(np.frombuffer(frame_buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED) for frame_buf in inp[index]]
        groundtruth_img = 255 * (gt[index].astype(np.uint8))

        return input_img, groundtruth_img

    # inputs in the form:
    # input_img: BGR, (len = D) H x W x C = ? x ? x 3, [0, 255] unit8
    # groundtruth_img: Binary, H x W = ? x ?, [False, True] bool
    def _img_to_tensor(self, input_img, groundtruth_img):
        # outputs in the form: 
        # input_tensor: BGR, C x D x H x W  = 3 x ? x ?, [0, 255] uint8 
        # groundtruth_tensor: Binary, C x D x H x W = 1 x 1 x ? x ?, [False, True] bool
        input_img =  np.stack(input_img, axis=0)
        input_tensor = torch.from_numpy(input_img).permute(3,0,1,2)
        groundtruth_tensor = torch.from_numpy(groundtruth_img).unsqueeze(0).unsqueeze(0).byte()
        return input_tensor, groundtruth_tensor

    # inputs in the form: 
    # input_tensor: BGR, H x W x C = ? x ? x 3, [0, 255] unit8
    # groundtruth_tensor: Binary, C x H x W = 1 x ? x ?, [False, True] bool
    def _tensor_to_img(self, input_tensor, groundtruth_tensor):
        # outputs in the form:
        # input_img: BGR, H x W x C = ? x ? x 3, [0, 255] unit8
        # groundtruth_img: Binary, H x W = ? x ?, [0, 255] uint8
        input_img = input_tensor.permute(1,2,0).numpy()
        groundtruth_img = groundtruth_tensor.squeeze(0).byte().numpy()
        return input_img, groundtruth_img

    ## don't need these for videos ## 
    def _get_input_iter(self, input_folder):
        pass 
    def _label_equiv(self, img_stem):
        pass

class LIRIS_dataset(ImageDataset):

    # Dataset info
    _folder_name = "VidSequences/LIRIS" # name of folder containing dataset in data_root (which contains folder with ALL datasets)
    _inp_folder_names = [] 
    _gt_folder_names = [] 

    # Dataset Statistics (BGR order) 
    _mean = FSD_Dataset._mean
    _std = FSD_Dataset._std

    def __init__(self, frame_num=5, **kwargs):
        self._file_name = f"LIRIS_Data_{frame_num}.npz"
        super().__init__(**kwargs) 

    # return the pandas dataframe containing raw/groundtruth files paths 
    def _load_data(self):

        # unpack dataset attributes
        data_file = self.data_file

        # create dataframe 
        if not Path(data_file).exists():
            print("Provide the data file for the LIRIS dataset")
            raise ValueError

        # load in existing datafile (will take some to load into RAM)
        loaded = np.load(data_file)
        return loaded['inp'], loaded['gt']

    # return input/label numpy image pair at index 
    def _get_images(self, index):

        # unpack dataset attributes
        inp, gt = self.inp, self.gt

        input_img = [cv2.imdecode(np.frombuffer(frame_buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED) for frame_buf in inp[index]]
        groundtruth_img = 255 * (gt[index].astype(np.uint8))

        return input_img, groundtruth_img

    # inputs in the form:
    # input_img: BGR, (len = D) H x W x C = ? x ? x 3, [0, 255] unit8
    # groundtruth_img: Binary, H x W = ? x ?, [False, True] bool
    def _img_to_tensor(self, input_img, groundtruth_img):
        # outputs in the form: 
        # input_tensor: BGR, C x D x H x W  = 3 x ? x ?, [0, 255] uint8 
        # groundtruth_tensor: Binary, C x D x H x W = 1 x 1 x ? x ?, [False, True] bool
        input_img =  np.stack(input_img, axis=0)
        input_tensor = torch.from_numpy(input_img).permute(3,0,1,2)
        groundtruth_tensor = torch.from_numpy(groundtruth_img).unsqueeze(0).unsqueeze(0).byte()
        return input_tensor, groundtruth_tensor

    # inputs in the form: 
    # input_tensor: BGR, H x W x C = ? x ? x 3, [0, 255] unit8
    # groundtruth_tensor: Binary, C x H x W = 1 x ? x ?, [False, True] bool
    def _tensor_to_img(self, input_tensor, groundtruth_tensor):
        # outputs in the form:
        # input_img: BGR, H x W x C = ? x ? x 3, [0, 255] unit8
        # groundtruth_img: Binary, H x W = ? x ?, [0, 255] uint8
        input_img = input_tensor.permute(1,2,0).numpy()
        groundtruth_img = groundtruth_tensor.squeeze(0).byte().numpy()
        return input_img, groundtruth_img

    ## don't need these for videos ## 
    def _get_input_iter(self, input_folder):
        pass 
    def _label_equiv(self, img_stem):
        pass