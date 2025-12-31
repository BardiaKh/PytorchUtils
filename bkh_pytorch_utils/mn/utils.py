from typing import List, Tuple, Dict, Union

import os
import copy
import shutil
import subprocess
from PIL import Image as Img
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

import numpy as np
import monai as mn

import timm
import torch

def empty_monai_cache(cache_dir:str, subsets = ["train", "val", "test"]) -> None:
    # Create an empty directory for rsync trick
    empty_dir = "/tmp/empty_dir_for_rsync"
    if not os.path.exists(empty_dir):
        os.mkdir(empty_dir)

    for subset in subsets:
        subset_path = os.path.join(cache_dir, subset)
        if os.path.exists(subset_path):
            # Use rsync to quickly delete files
            subprocess.call(['rsync', '-a', '--delete', f'{empty_dir}/', subset_path])
            print(f"MONAI's {subset} cache directory removed successfully!")
        else:
            print(f"MONAI's {subset} cache directory does not exist!")

    # Cleanup temporary empty directory
    os.rmdir(empty_dir)
    
def cache_dataset(dataset, desc, num_workers=32):
    assert isinstance(dataset, mn.data.PersistentDataset), "Dataset must be an instance of MONAI's PersistentDataset"
    
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(lambda i=i: dataset[i], i) for i in range(len(dataset))]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=desc):
            pass

class EnsureGrayscaleD(mn.transforms.MapTransform):
    def __init__(self, keys:List[str]) -> None:
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img=d[key]
            
            acceptable_channels = [1, 2, 3, 4]

            if len(img.shape)==2: # Back & White
                img = img.unsqueeze(0)
            
            if img.shape[0] in acceptable_channels: # Channel First
                img = img.mean(dim = 0)
                img = img.unsqueeze(0)
            
            if img.shape[-1] in acceptable_channels: # Channel Last
                img = img.mean(dim = -1)
                img = img.unsqueeze(0)
            
            if len(img.shape)==3 and img.shape[0]!=1 and img.shape[-1]!=1: #3D image
                img = img.unsqueeze(0)

            d[key]=img
        return d

class ConvertToPIL(mn.transforms.Transform):
    def __init__(self, mode:str="RGB", transpose=True) -> None:
        super().__init__()
        self.mode=mode.upper()
        self.transpose=transpose

    def __call__(self, data):
        img=copy.deepcopy(data)
        img=self.to_numpy(img)

        if self.mode=="RGB":
            if len(img.shape)==2:
                img = np.expand_dims(img,axis=-1)
                img = np.concatenate([img,img,img], axis=2)
            elif len(img.shape)==3:
                if img.shape[-1]==4: # Fixing RGBA
                    img = img[:,:,:3]
                elif img.shape[0]==4:
                    img = img[:3,:,:]
                
                if self.transpose:
                    if img.shape[0]==1 or img.shape[0]==3:
                        img = img.transpose(2,1,0)
                    elif img.shape[-1]==1 or img.shape[-1]==3:
                        img = img.transpose(1,0,2)

                if img.shape[-1]==1:
                    img = np.concatenate([img,img,img], axis=2)

        if self.mode=="L": #n-dims should be 2
            if len(img.shape)==3:
                if img.shape[-1]==3 or img.shape[-1]==4: # Channel Last
                    img = np.mean(img, axis=-1)
                elif img.shape[0]==3 or img.shape[0]==4: # Channel First
                    img = np.mean(img, axis=0)
                elif img.shape[0]==1:
                    img = img[0,:,:]
                elif img.shape[-1]==1:
                    img = img[:,:,0]

                if self.transpose:
                    img = img.transpose(1,0)

        img = Img.fromarray(img.astype('uint8'), self.mode)
            
        return img
    
    def to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.numpy()
        elif isinstance(data, np.ndarray):
            return data
    
class RandAugD(mn.transforms.MapTransform):
    def __init__(self, keys:List[str], pil_conversion_mode:str = "RGB", m:int=9, n:int=2, mstd:float=0.5, convert_to_numpy:bool=True) -> None:
        super().__init__(keys)
        self.converter = ConvertToPIL(mode=pil_conversion_mode)
        self.convert_to_numpy = convert_to_numpy
        timm.data.auto_augment._RAND_TRANSFORMS  = [
            'AutoContrast',
            'Equalize',
            #'Invert',
            'Rotate',
            #'Posterize',
            #'Solarize',
            #'SolarizeAdd',
            #'Color',
            'Contrast',
            'Brightness',
            'Sharpness',
            'ShearX',
            'ShearY',
            'TranslateXRel',
            'TranslateYRel',
        ]
        self.augmentor = timm.data.auto_augment.rand_augment_transform(config_str=f"rand-n{n}-m{m}-mstd{mstd}", hparams={})
        
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            img = self.converter(img)
            img = self.augmentor(img)
            
            if self.convert_to_numpy:
                img = np.array(img)
                
            d[key] = img
        return d
