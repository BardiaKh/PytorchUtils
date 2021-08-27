import os
import torch
from tabulate import tabulate
import pytorch_lightning as pl

class BKhModule(pl.LightningModule):
    def __init__(self, collate_fn=None, sampler=None, train_ds=None, val_ds=None, dl_workers=-1, batch_size=None):
        super().__init__()
        self.collate_fn = collate_fn
        self.batch_size=batch_size

        self.total_steps = None
        self.last_stepped_step=-1

        self.dl_workers = min(os.cpu_count()*2,8) if dl_workers==-1 else dl_workers

        self.train_dl = None
        self.val_dl = None

        if train_ds is not None:
            self.set_train_dataset(train_ds, sampler=sampler)

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

    def set_total_steps(self,steps=None,last_stepped_step=None):
        if steps is not None:
            self.total_steps=steps
        if last_stepped_step is not None:
            self.last_stepped_step=last_stepped_step

    def forward(self,x):
        return self.model(x)

    def set_train_dataset(self, ds, sampler=None, rtn=False):
        self.train_dl=torch.utils.data.DataLoader(ds, batch_size=self.batch_size, sampler=sampler, shuffle=True if sampler is None else False, num_workers=self.dl_workers, collate_fn=self.collate_fn, pin_memory=False, drop_last=True, prefetch_factor=1)
        if rtn:
            return self.train_dl

    def set_val_dataset(self, ds, rtn=False):
        self.val_dl=torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.dl_workers, collate_fn=self.collate_fn, pin_memory=False, drop_last=False, prefetch_factor=1)
        if rtn:
            return self.val_dl

    def train_dataloader(self):
        if self.train_dl is None:
            raise Exception("Use 'set_train_dataset' to set the training dataset.")
        else:
            return self.set_train_dataset(self.train_dl.dataset, self.train_dl.sampler,rtn=True)

    def val_dataloader(self):
        if self.val_dl is None:
            raise Exception("Use 'set_val_dataset' to set the validation dataset.")
        else:
            return self.set_val_dataset(self.val_dl.dataset,rtn=True)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items