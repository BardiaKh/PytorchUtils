from typing import List, Tuple, Dict, Union

import os
import copy
import shutil
from skimage.filters import unsharp_mask, meijering, sato, scharr, hessian
from skimage.exposure import equalize_hist, equalize_adapthist

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
    def __init__(self, include:List[str]) -> None:
        super().__init__()
        self.include=include

    def __call__(self, data):
        data_copy=copy.deepcopy(data)
        for key in data:
            if key not in self.include:
                data_copy.pop(key,None)
        return data_copy

class EnsureGrayscaleD(mn.transforms.Transform):
    def __init__(self, keys:List[str]) -> None:
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

class ApplySkimageFilterD(mn.transforms.RandomizableTransform):
    def __init__(self, keys:List[str], filter_name:str, config:dict, prob:float=0.1) -> None:
        super().__init__(prob)
        self.keys=keys
        self.filter_name=filter_name
        self.config=config

    def __call__(self, data):
        super().randomize(None)

        if not self._do_transform:
            return data

        data_copy=copy.deepcopy(data)
        for key in data:
            if key in self.keys:
                img=data[key].copy()

                if self.filter_name == 'equalize_hist':
                    img = equalize_hist(img, nbins= self.config.get('nbins',256), mask=self.config.get('mask',None))

                if self.filter_name == 'equalize_adapthist':
                    img = equalize_adapthist(img, kernel_size=self.config.get('kernel_size',None), clip_limit=self.config.get('clip_limit',0.01), nbins=self.config.get('nbins',256))

                if self.filter_name == 'unsharp_mask':
                    img = unsharp_mask(img, radius=self.config.get('radius',5), amount=self.config.get('amount',2))

                if self.filter_name == 'meijering':
                    img = meijering(img, sigmas=self.config.get('sigmas',range(1, 10, 2)), alpha=self.config.get('alpha',None), black_ridges=self.config.get('black_ridges',True), mode=self.config.get('mode','reflect'), cval=self.config.get('cval',0))

                if self.filter_name == 'sato':
                    img = sato(img, sigmas=self.config.get('sigmas',range(1, 10, 2)), black_ridges=self.config.get('black_ridges',True), mode=self.config.get('mode','reflect'), cval=self.config.get('cval',0))

                if self.filter_name == 'scharr':
                    img = scharr(img, mask=self.config.get('mask',None), axis=self.config.get('axis',None), mode=self.config.get('mode','reflect'), cval=self.config.get('cval',0))

                if self.filter_name == 'hessian':
                    img = hessian(img, sigmas=self.config.get('sigmas',range(1, 10, 2)), scale_range=self.config.get('scale_range',None), scale_step=self.config.get('scale_step',None), alpha=self.config.get('alpha',0.5), beta=self.config.get('beta',0.5), gamma=self.config.get('cval',15), black_ridges=self.config.get('black_ridges',True), mode=self.config.get('mode','reflect'), cval=self.config.get('cval',0))

                data_copy[key]=img
        return data_copy
