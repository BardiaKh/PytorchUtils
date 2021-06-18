from typing import List, Tuple, Dict, Union

import os
import random
import torch

import numpy as np
from tqdm.auto import tqdm
import monai as mn
import pytorch_lightning as pl


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
