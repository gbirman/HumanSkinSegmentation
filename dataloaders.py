import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler

class CustomDataLoader: 

    def __init__(self, dataset, batch_size=1, validation_split=None, split_file=None, shuffle_dataset=False, random_seed=None, pin_memory=False, num_workers=4):
        self.dataset = dataset 
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle_dataset = shuffle_dataset
        self.random_seed = random_seed
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.split_file = split_file
    
    class CustomBatch:
        def __init__(self, data):
            transposed_data = list(zip(*data))
            self.inp = torch.cat(transposed_data[0], 0)
            self.tgt = torch.cat(transposed_data[1], 0)
            
        # custom memory pinning method on custom type
        def pin_memory(self):
            self.inp = self.inp.pin_memory()
            self.tgt = self.tgt.pin_memory()
            return self

    def collate_wrapper(self, batch):
        return self.CustomBatch(batch)

    def getLoader(self, mode='test'):

        if self.random_seed:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Creating data indices for training and validation splits:
        if mode == 'train' or mode == 'val':

            if self.split_file:
                loaded = np.load(self.split_file, allow_pickle=True)
                train_indices, test_indices = loaded['train'], loaded['test']
                dataset_size = len(test_indices)
                split = int(np.floor(self.validation_split * dataset_size))
                val_indices = test_indices[:split]
            elif self.validation_split:
                dataset_size = len(self.dataset)
                indices = list(range(dataset_size))
                split = int(np.floor(self.validation_split * dataset_size))
                if self.shuffle_dataset:
                    np.random.shuffle(indices)
                train_indices, val_indices = indices[split:], indices[:split]
            
            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            if mode == 'train':
                return DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler, collate_fn=self.collate_wrapper, pin_memory=self.pin_memory, num_workers=self.num_workers)
            elif mode == 'val':
                return DataLoader(self.dataset, batch_size=1, sampler=valid_sampler, collate_fn=self.collate_wrapper, pin_memory=self.pin_memory, num_workers=self.num_workers)  
        elif mode == 'test':
            if self.split_file:
                loaded = np.load(self.split_file, allow_pickle=True)
                test_indices = loaded['test']
                dataset_size = len(test_indices)
                split = int(np.floor(self.validation_split * dataset_size))
                test_indices = test_indices[split:]
                sampler = SubsetRandomSampler(test_indices) # doesn't really matter that it's random for holistic evaluation but can be modified to sequential for easier debugging
            elif self.shuffle_dataset:
                sampler = RandomSampler(self.dataset)
            else:
                sampler = SequentialSampler(self.dataset)
                
            return DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler, collate_fn=self.collate_wrapper, pin_memory=self.pin_memory, num_workers=self.num_workers)
        else:
            print("Invalid mode.")
            quit()