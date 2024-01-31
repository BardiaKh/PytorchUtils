from typing import Iterator, Sequence, Union, List

from .cm_helper import pretty_plot_confusion_matrix

import os
import math
import copy
import random
import pickle
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.cuda.amp import autocast
import functools

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import monai as mn
import lightning as pl

from string import ascii_uppercase
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold

def seed_all(seed:int) -> None:
    """Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): seed to use
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    mn.utils.misc.set_determinism(seed=seed)
    pl.seed_everything(seed,workers=True)

def get_data_stats(dataset:torch.utils.data.Dataset, img_key:str, num_channels:int = 1, num_workers: int = 4) -> None:
    #########################################################################################################
    ### Adapted from: https://github.com/Nikronic/CoarseNet/blob/master/utils/preprocess.py#L142-L200
    #########################################################################################################
    """
    Compute dataset mean and standard deviation in an online manner using DataLoader and multiprocessing.

    Args:
    - dataset: a PyTorch Dataset object
    - img_key: the key to extract the image tensor from the sample
    - num_channels: number of image channels
    - num_workers: number of processes to use for parallel computation

    Returns: None (prints mean and std)
    """
    def _compute_image_stats(image):
        """Compute the sum, squared sum, and number of pixels for a single image."""
        sum_ = image.sum()
        sum_of_square = (image**2).sum()
        nb_pixels = torch.prod(torch.tensor(image.shape)).item()
        
        return sum_, sum_of_square, nb_pixels

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # Initializations
    cnt = 0
    fst_moment = torch.zeros(num_channels)
    snd_moment = torch.zeros(num_channels)

    # Using tqdm to show progress for data loading
    for sample in tqdm(dataloader, desc="Computing mean"):
        sum_, sum_of_square, nb_pixels = _compute_image_stats(sample[img_key])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean = fst_moment
    std = torch.sqrt(snd_moment - fst_moment**2)

    print("Final Mean:", mean)
    print("Final Std:", std)

def one_hot_encode(true_labels: torch.Tensor, num_classes: int, smoothing=0.0):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), num_classes))
    if label_shape == true_labels.size():
        with torch.no_grad():
            true_dist = torch.where(true_labels==1.0, confidence, smoothing)
    else:
        with torch.no_grad():
            true_dist = torch.empty(size=label_shape, device=true_labels.device)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)

    return true_dist

def plot_confusion_matrix(preds:np.array, targets:np.array, columns:list=None, annot:bool=True, cmap:str="Oranges",
      fmt:str='.2f', fz:int=13, lw:float=0.5, cbar:bool=False, figsize:list=[9,9], show_null_values:int=1, pred_val_axis:str='x', save_name=None):

    if columns is None:
        columns = ['Class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(targets))]]

    matrix = confusion_matrix(targets, preds)
    df_cm = pd.DataFrame(matrix, index=columns, columns=columns)

    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, annot=annot, fmt=fmt, lw=lw, cbar=cbar, show_null_values=show_null_values, pred_val_axis=pred_val_axis, save_name = save_name)

def load_weights(model: torch.nn.Module, weight_path: str = None):
    weights = torch.load(weight_path)
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    model_dict.update(weights) 
    model.load_state_dict(model_dict)

    return model


def add_weight_decay(models: Union[torch.nn.Module, List[torch.nn.Module]], weight_decay:float=1e-5, skip_list:list=[]):
    #########################################################################################################
    ### Adapted from: https://github.com/rwightman/pytorch-image-models/tree/master/timm
    #########################################################################################################
    if not isinstance(models, list):
        models = [models]
    
    decay = []
    no_decay = []
    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
    ]
        
def is_notebook_running():
    try:
        shell = get_ipython().__class__
        if 'google.colab._shell.Shell' in str(shell):
            return True
        if shell.__name__ == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell.__name__ == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def split_data(df: pd.DataFrame, n_splits: int, y_column: str=None, group_column:str=None, fold_column: str="Fold", shuffle=False, random_state=None):
    df = df.copy()

    if random_state is not None:
        shuffle = True
    elif shuffle and random_state is None:
        random_state = 42

    if y_column is None and group_column is None:
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        print("Using simple KFold split...")
    elif y_column is not None and group_column is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        print("Using StratifiedKFold split...")
    elif y_column is None and group_column is not None:
        splitter = GroupKFold(n_splits=n_splits)
        if shuffle:
            print("GroupKFold does not support shuffle. Setting shuffle to False...")
        print("Using GroupKFold split...")
    elif y_column is not None and group_column is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        print("Using StratifiedGroupKFold split...")

    df[fold_column] = 0

    for fold_idx, (train_index, val_index) in enumerate(splitter.split(df, y=df[y_column].tolist() if y_column is not None else None, groups=df[group_column].tolist() if group_column is not None else None)):
        df.loc[val_index,fold_column]=fold_idx

    return df
    
class ExhaustiveWeightedRandomSampler(WeightedRandomSampler):
    #########################################################################################################
    ### Adapted from: https://github.com/louis-she/exhaustive-weighted-random-sampler
    #########################################################################################################
    """ExhaustiveWeightedRandomSampler behaves pretty much the same as WeightedRandomSampler
    except that it receives an extra parameter, exaustive_weight, which is the weight of the
    elements that should be sampled exhaustively over multiple iterations.
    This is useful when the dataset is very big and also very imbalanced, like the negative
    sample is way more than positive samples, we want to over sample positive ones, but also
    iterate over all the negative samples as much as we can.
    Args:
        weights (sequence): a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        exaustive_weight (int): which weight of samples should be sampled exhaustively,
            normally this is the one that should not been over sampled, like the lowest
            weight of samples in the dataset.
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        exaustive_weight=1,
        generator=None,
    ) -> None:
        super().__init__(weights, num_samples, True, generator)
        self.all_indices = torch.tensor(list(range(num_samples)))
        self.exaustive_weight = exaustive_weight
        self.weights_mapping = torch.tensor(weights) == self.exaustive_weight
        self.remaining_indices = torch.tensor([], dtype=torch.long)

    def get_remaining_indices(self) -> torch.Tensor:
        remaining_indices = self.weights_mapping.nonzero().squeeze()
        return remaining_indices[torch.randperm(len(remaining_indices))]

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=self.generator
        )
        exaustive_indices = rand_tensor[
            self.weights_mapping[rand_tensor].nonzero().squeeze()
        ]
        while len(exaustive_indices) > len(self.remaining_indices):
            self.remaining_indices = torch.cat(
                [self.remaining_indices, self.get_remaining_indices()]
            )
        yield_indexes, self.remaining_indices = (
            self.remaining_indices[: len(exaustive_indices)],
            self.remaining_indices[len(exaustive_indices) :],
        )
        rand_tensor[
            (rand_tensor[..., None] == exaustive_indices).any(-1).nonzero().squeeze()
        ] = yield_indexes
        yield from iter(rand_tensor.tolist())

def autocast_inference(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            with autocast():
                return func(*args, **kwargs)
    return wrapper
