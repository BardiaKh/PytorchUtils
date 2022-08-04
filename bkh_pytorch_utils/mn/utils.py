from typing import List, Tuple, Dict, Union

import os
import copy
import shutil
from PIL import Image as Img

import numpy as np
import monai as mn

import timm
import torch

def empty_monai_cache(cache_dir:str) -> None:
    if os.path.exists(cache_dir+"/train"):
        shutil.rmtree(cache_dir+"/train")
        print("MOANI's train cache directory removed successfully!")

    if os.path.exists(cache_dir+"/val"):
        shutil.rmtree(cache_dir+"/val")
        print("MOANI's validation cache directory removed successfully!")

    if os.path.exists(cache_dir+"/test"):
        shutil.rmtree(cache_dir+"/test")
        print("MOANI's test cache directory removed successfully!")


class EnsureGrayscaleD(mn.transforms.Transform):
    def __init__(self, keys:List[str]) -> None:
        super().__init__()
        self.keys=keys

    def __call__(self, data):
        data_copy=copy.deepcopy(data)
        for key in data:
            if key in self.keys:
                img=data[key].copy()

                if len(img.shape)==2: # Back & White
                    img=np.expand_dims(img,axis=0)
                elif img.shape[0]==1 or img.shape[0]==3: # Channel First
                    img=np.mean(img,axis=0)
                    img=np.expand_dims(img,axis=0)
                elif img.shape[-1]==1 or img.shape[-1]==3: # Channel Last
                    img=np.mean(img,axis=2)
                    img=np.expand_dims(img,axis=0)

                data_copy[key]=img
        return data_copy

class TransposeD(mn.transforms.Transform):
    def __init__(self, keys:List[str], indices:tuple) -> None:
        super().__init__()
        self.keys=keys
        self.transposer=mn.transforms.Transpose(indices)

    def __call__(self, data):
        data_copy=copy.deepcopy(data)
        for key in data:
            if key in self.keys:
                img=data[key].copy()
                data_copy[key]=self.transposer(img)
        return data_copy

class ConvertToPIL(mn.transforms.Transform):
    def __init__(self, mode:str="RGB") -> None:
        super().__init__()
        self.mode=mode.upper()

    def __call__(self, data):
        img=data.copy()
        img=self.to_numpy(img)

        if self.mode=="RGB":
            if len(img.shape)==2:
                img = np.expand_dims(img,axis=-1)
                img = np.concatenate([img,img,img], axis=2)
            elif len(img.shape)==3:
                if img.shape[0]==1 or img.shape[0]==3:
                    img = img.transpose(1,2,0)

                if img.shape[-1]==1:
                    img = np.concatenate([img,img,img], axis=2)

        if self.mode=="L":
            if len(img.shape)==2:
                img = np.expand_dims(img,axis=-1)
            elif len(img.shape)==3:
                if img.shape[-1]==3:
                    img = np.mean(img, axis=-1)

        img = Img.fromarray(img.astype('uint8'), self.mode)
            
        return img
    
    def to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.numpy()
        elif isinstance(data, np.ndarray):
            return data
    
class RandAugD(mn.transforms.RandomizableTransform):
    def __init__(self, keys:List[str], pil_conversion_mode:str = "RGB", m:int=9, n:int=2, mstd:float=0.5, convert_to_numpy:bool=True) -> None:
        super().__init__()
        self.keys=keys
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
        data_copy=copy.deepcopy(data)
        for key in data:
            if key in self.keys:
                img = data[key].copy()
                img = self.converter(img)
                img = self.augmentor(img)
                
                if self.convert_to_numpy:
                    img = np.array(img)
                    
                data_copy[key] = img
        return data_copy
