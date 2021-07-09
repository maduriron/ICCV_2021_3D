import os
from PIL import Image
import re

import pandas as pd
import torch
import json

import numpy as np
from PIL import Image
import tqdm
from .gtransforms import *

import torchvision.transforms as transforms
from PIL import Image
import PIL
import cv2
import os

import SimpleITK as sitk
import random

from torch.utils.data import Dataset, DataLoader
  
def kinetics_mean_std():
    mean = [114.75, 114.75, 114.75]
    std = [57.375, 57.375, 57.375]
    return mean, std

def clip_transform(split, max_len, shape=256, transform_list=None):
    mean, std = kinetics_mean_std()
    if transform_list is None:
        if split == 'train':
            transform = transforms.Compose([
                            #GroupShearY(),
                            #GroupShearX(),
                            #GroupCutOut(),
                            GroupRandomHorizontalFlip(), # add back
                            GroupRandomVerticalFlip(), # add back
                            GroupContrast(), # add back
                            GroupGaussianBlur(), # add back
                            GroupColor(), # add back
                            # GroupResize(size=(224, 224)), # DON'T ADD BACK
                            ToTensor(), # add back
                            GroupNormalize(mean, std), # add back
                            LoopPad(max_len), # add back
                        ])
        elif split == 'val':
            transform = transforms.Compose([
                            # GroupResize(size=(224, 224)),
                            ToTensor(),
                            GroupNormalize(mean, std),
                            LoopPad(max_len),
                ])
        elif split=='3crop' or split == 'test':
            transform = transforms.Compose([
                    # GroupResize(size=(256, 256)),
                    ToTensor(),
                    GroupNormalize(mean, std),
                    LoopPad(max_len),
                ]) 
    else:
        transform_list.append(ToTensor())
        transform_list.append(GroupNormalize(mean, std))
        transform_list.append(LoopPad(max_len))
        transform = transforms.Compose(transform_list)
        
    return transform 


class CTDataset(Dataset):
    def __init__(self, root, fold_id=0, fold_splitter=None, transforms=None,
                replacer=None, prepath=None, clip_len=0, split=None):
        """
        params:
        root := directory where data is hold
        fold_id := id number of split for current training process
        fold_splitter := {"fold_id0": {"paths": [<<list of paths>>], "metadata": [<<list of metadata>>]},
                        "fold_id1": {"paths": [<<list of paths>>], "metadata": [<<list of metadata>>]},
                        ...
                        }
        transforms := transforms that should be done for current data
        """
        super(CTDataset, self).__init__()
        self.root = root
        self.fold_id = fold_id
        self.fold_splitter = fold_splitter
        self.transforms = transforms
        self.replacer = replacer
        self.prepath = prepath
        self.clip_len = clip_len
        self.split = split
        self.clip_transform = clip_transform(self.split, self.clip_len, self.transforms)	
        self.loader = lambda fl: Image.open('%s/%s'%(self.root, fl)).convert('RGB')	

    def __len__(self):
        return len(self.fold_splitter[self.fold_id]["paths"])

    def load_ct(self, path, offset, end, leap, remainder=None):
        imgs = []
        v_len = len(os.listdir(path)) # number of imags in the dir
        prob = random.uniform(0, 1)
        if prob <= 0.5: # forward
            for i in range(offset, end, leap):
                imgs.append(Image.open(os.path.join(path, str(i) + ".jpg")).convert("RGB"))
            return imgs
        else: # backward
            for i in range(end - 1, offset - 1, -leap):
                imgs.append(Image.open(os.path.join(path, str(i) + ".jpg")).convert("RGB"))
            return imgs

    def sample(self, frames, video_path):
        if frames > self.clip_len:
            if self.split == 'train': 
                if frames <= self.clip_len:
                    leap = 1
                elif frames <= 2 * self.clip_len: 
                    leap = 2
                elif frames <= 3 * self.clip_len:
                    leap = 3
                elif frames <= 4 * self.clip_len:
                    leap = 4
                else:
                    leap = 5
                offset = np.random.randint(0, leap)
                # end = offset + leap * self.clip_len 
                end = frames
            elif self.split == 'val':
                if frames <= 2 * self.clip_len:
                    leap = 1
                elif frames <= 4 * self.clip_len: 
                    leap = 2
                else:
                    leap = 3
                offset = np.random.randint(0, leap)
                end = frames
        else:
            leap = 1
            offset = 0
            end = frames
        
        imgs = self.load_ct(video_path, offset, end, leap)
        
        return imgs
    
    def __getitem__(self, id):
        path = self.fold_splitter[self.fold_id]["paths"][id]
        path = path.replace(self.prepath, self.replacer)
        metadata_label = self.fold_splitter[self.fold_id]["metadata"][id]
        metadata_frames = self.fold_splitter[self.fold_id]["frames"][id]
        path = os.path.join(self.root, path) # path to directory of all imgs

        frames = self.sample(metadata_frames, path)
        frames = self.clip_transform(frames) # (T, 3, 224, 224)
        frames = frames.permute(1, 0, 2, 3) # (3, T, 224, 224
        
        # instance = {'frames': frames, 'label': metadata_label}
        return frames, metadata_label#instance

class CTDatasetTestSimple(Dataset):
    def __init__(self, root, fold_id=0, fold_splitter=None, transforms=None,
                replacer=None, prepath=None, clip_len=0, split=None):
        """
        params:
        root := directory where data is hold
        fold_id := id number of split for current training process
        fold_splitter := {"fold_id0": {"paths": [<<list of paths>>], "metadata": [<<list of metadata>>]},
                        "fold_id1": {"paths": [<<list of paths>>], "metadata": [<<list of metadata>>]},
                        ...
                        }
        transforms := transforms that should be done for current data
        """
        super(CTDatasetTestSimple, self).__init__()
        self.root = root
        self.fold_id = fold_id
        self.fold_splitter = fold_splitter
        self.transforms = transforms
        self.replacer = replacer
        self.prepath = prepath
        self.clip_len = clip_len
        self.split = split
        self.clip_transform = clip_transform(self.split, self.clip_len, self.transforms)	
        self.loader = lambda fl: Image.open('%s/%s'%(self.root, fl)).convert('RGB')	

    def __len__(self):
        return len(self.fold_splitter[self.fold_id]["paths"])

    def load_ct(self, path, offset, end, leap, remainder=None):
        imgs = []
        v_len = len(os.listdir(path)) # number of imags in the dir
        prob = random.uniform(0, 1)
        if prob <= 1: # forward
            for i in range(offset, end, leap):
                imgs.append(Image.open(os.path.join(path, str(i) + ".jpg")).convert("RGB"))
            return imgs
        else: # backward
            for i in range(end - 1, offset - 1, -leap):
                imgs.append(Image.open(os.path.join(path, str(i) + ".jpg")).convert("RGB"))
            return imgs

    def sample(self, frames, video_path):
        if frames > self.clip_len:
            if self.split == 'train': 
                if frames <= self.clip_len:
                    leap = 1
                elif frames <= 2 * self.clip_len: 
                    leap = 2
                elif frames <= 3 * self.clip_len:
                    leap = 3
                elif frames <= 4 * self.clip_len:
                    leap = 4
                else:
                    leap = 5
                offset = np.random.randint(0, leap)
                # end = offset + leap * self.clip_len 
                end = frames
            elif self.split == 'val':
                if frames <= 2 * self.clip_len:
                    leap = 1
                elif frames <= 4 * self.clip_len: 
                    leap = 2
                else:
                    leap = 3
                offset = np.random.randint(0, leap)
                end = frames
            elif self.split == 'test':
                offset = 0
                end = frames
                leap = 1
        else:
            leap = 1
            offset = 0
            end = frames
        
        imgs = self.load_ct(video_path, offset, end, leap)
        
        return imgs
    
    def __getitem__(self, id):
        path = self.fold_splitter[self.fold_id]["paths"][id]
        path = path.replace(self.prepath, self.replacer)
        metadata_label = self.fold_splitter[self.fold_id]["metadata"][id]
        metadata_frames = self.fold_splitter[self.fold_id]["frames"][id]
        path = os.path.join(self.root, path)

        frames = self.sample(metadata_frames, path)
        frames = self.clip_transform(frames) 
        frames = frames.permute(1, 0, 2, 3)  
        return frames, metadata_label, path