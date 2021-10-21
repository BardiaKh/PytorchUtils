import os
import torch
from torch.utils.data.distributed import DistributedSampler
from tabulate import tabulate
import pytorch_lightning as pl

from .ddp_helper import DistributedProxySampler

class BKhModule(pl.LightningModule):
    def __init__(self, collate_fn=None, sampler=None, ddp_sampler=False, train_ds=None, val_ds=None, dl_workers=-1, batch_size=None):
        super().__init__()
        self.collate_fn = collate_fn
        self.batch_size = batch_size

        self.total_steps = None
        self.last_stepped_step = -1

        self.dl_workers = min(os.cpu_count()*2,8) if dl_workers==-1 else dl_workers

        self.train_ds = None
        self.val_ds = None

        self.train_dl = None
        self.val_dl = None

        self.sampler = sampler
        self.ddp_sampler = ddp_sampler

        if train_ds is not None:
            self.set_train_dataset(train_ds)

        if val_ds is not None:
            self.set_val_dataset(val_ds)

    def stats(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        freezed_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

        print(tabulate([["Learnable",trainable_params],["Freezed",freezed_params],["Total",trainable_params+freezed_params]],headers=['Params','Count'],tablefmt='orgtbl', colalign=("left","right")))
        print("\n")
        print(f"Model Size: {self.model_size:.02F}MB")

    def get_best_checkpoint_path(self):
        if self.trainer is None:
            raise Exception("Use 'trainer.fit(Module)' first to allocated the module to the trainer.")

        callbacks = self.trainer.callbacks
        for callback in callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                best_weight_path=callback.best_model_path
                if best_weight_path.endswith(".ckpt"):
                    return best_weight_path
                else:
                    raise Exception("No valid checkpoint file is found.")

                break
        return None

    def set_total_steps(self,steps=None,last_stepped_step=-1):
        if steps is not None:
            self.total_steps = steps
        if last_stepped_step is not None:
            self.last_stepped_step = last_stepped_step

    def forward(self,x):
        return self.model(x)

    def set_train_dataset(self, ds):
        self.train_ds = ds

    def set_val_dataset(self, ds):
        self.val_ds = ds

    def train_dataloader(self):
        if self.train_ds is None:
            raise Exception("Use the 'set_train_dataset' method to set the training dataset.")
        else:
            if self.ddp_sampler:
                if self.sampler is None:
                    instance_sampler = DistributedSampler(self.train_ds)
                else:
                    instance_sampler = DistributedProxySampler(self.sampler)
            else:
                instance_sampler = self.sampler

            self.train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, sampler=instance_sampler, shuffle=True if instance_sampler is None else False, num_workers=self.dl_workers, collate_fn=self.collate_fn, pin_memory=False, drop_last=True, prefetch_factor=1)
            return self.train_dl

    def val_dataloader(self):
        if self.val_ds is None:
            raise Exception("Use the 'set_val_dataset' method to set the validation dataset.")
        else:
            if self.ddp_sampler:
                instance_sampler = DistributedSampler(self.val_ds)
            else:
                instance_sampler = None

            self.val_dl = torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, sampler=instance_sampler, shuffle=True if instance_sampler is None else False, num_workers=self.dl_workers, collate_fn=self.collate_fn, pin_memory=False, drop_last=False, prefetch_factor=1)
            return self.val_dl

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items