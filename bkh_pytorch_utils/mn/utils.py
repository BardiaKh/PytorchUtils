from typing import List, Tuple, Dict, Union

import os
import copy
import shutil

import numpy as np
import monai as mn

def empty_monai_cache(cache_dir:str) -> None:
    if os.path.exists(cache_dir+"/train"):
        shutil.rmtree(cache_dir+"/train")
        print("MOANI's train cache directory removed successfully!")

    if os.path.exists(cache_dir+"/val"):
        shutil.rmtree(cache_dir+"/val")
        print("MOANI's validation cache directory removed successfully!")

class FilterKeys(mn.transforms.Transform):
    def __init__(self,include:List[str]) -> None:
        super().__init__()
        self.include=include

    def __call__(self, data):
        data_copy=copy.deepcopy(data)
        for key in data:
            if key not in self.include:
                data_copy.pop(key,None)
        return data_copy

class EnsureGrayscaleD(mn.transforms.Transform):
    def __init__(self,keys:List[str]) -> None:
        super().__init__()
        self.keys=keys

    def __call__(self, data):
        data_copy=copy.deepcopy(data)
        for key in data:
            if key in self.keys:
                img=data[key].copy()

                if len(img.shape)==2:
                    img=np.expand_dims(img,axis=0)
                elif img.shape[-1]>2:
                    img=np.mean(img,axis=2)
                    img=np.expand_dims(img,axis=0)

                data_copy[key]=img
        return data_copy

class TransposeD(mn.transforms.Transform):
    def __init__(self,keys:List[str], indices:tuple) -> None:
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
