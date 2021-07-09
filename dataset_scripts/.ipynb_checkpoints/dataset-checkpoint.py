import os
from PIL import Image
import re

from torch.utils.data import Dataset, DataLoader
    

class LeukemiaDataset(Dataset):
    def __init__(self, root, fold_id=0, fold_splitter=None, transforms=None):
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
        self.root = root
        self.fold_id = fold_id
        self.fold_splitter = fold_splitter
        self.transforms = transforms
        
    def __len__(self):
        return len(self.fold_splitter[self.fold_id]["paths"])

    def __getitem__(self, id):
        path = self.fold_splitter[self.fold_id]["paths"][id]
        metadata = self.fold_splitter[self.fold_id]["metadata"][id]
        
        path = os.path.join(self.root, path)
        img = Image.open(path).convert("RGB")
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, metadata["label"]
        

