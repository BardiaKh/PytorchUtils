from typing import List, Tuple, Dict, Union

from .cm_helper import pretty_plot_confusion_matrix

import os
import random
import torch

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import monai as mn
import pytorch_lightning as pl

from string import ascii_uppercase
from sklearn.metrics import confusion_matrix


def seed_all(seed:int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    mn.utils.misc.set_determinism(seed=seed)
    pl.seed_everything(seed,workers=True)

def get_data_stats(dataset:torch.utils.data.Dataset, img_key:str, dims:int = 1)->None:
    pixels_sum=torch.zeros(dims)
    pixels_count=torch.zeros(dims)
    sum_squared_err=torch.zeros(dims)

    for i,b in enumerate(tqdm(dataset)):
        image = b[img_key]
        pixels_sum = pixels_sum+image.sum((1,2))
        pixels_count = pixels_count+torch.tensor([image.shape[1]*image.shape[2]]*dims)

    mean = pixels_sum/pixels_count

    for i,b in enumerate(tqdm(dataset)):
        image = b[img_key].reshape(-1,dims)
        sum_squared_err = sum_squared_err + ((image - mean).pow(2)).sum()

    std = torch.sqrt(sum_squared_err / pixels_count)

    print("Final Mean:",mean)
    print("Final Std:",std)

def plot_confusion_matrix(preds:np.array, targets:np.array, columns:list=None, annot:bool=True, cmap:str="Oranges",
      fmt:str='.2f', fz:int=11, lw:float=0.5, cbar:bool=False, figsize:list=[8,8], show_null_values:int=0, pred_val_axis:str='col'):

    if columns is None:
        columns = ['Class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(targets))]]

    matrix = confusion_matrix(targets, preds)
    df_cm = pd.DataFrame(matrix, index=columns, columns=columns)

    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, annot=annot, fmt=fmt, lw=lw, cbar=cbar, show_null_values=show_null_values, pred_val_axis=pred_val_axis)
