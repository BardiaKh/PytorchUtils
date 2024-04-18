from typing import Optional, Union, Dict, Any
import os
import torch
import copy
from overrides import overrides
from tabulate import tabulate
import lightning as pl
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data.distributed import DistributedSampler
from .ddp_helper import DistributedProxySampler
try:
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
except ImportError:
    deepspeed = None

class BKhModule(pl.LightningModule):
    def __init__(self, collate_fn=None, val_collate_fn=None, train_sampler=None, val_sampler=None, ddp_sampler=False, train_ds=None, val_ds=None, dl_workers=-1, batch_size=None, val_batch_size=None, pin_memory=True, prefetch_factor=1, persistent_workers=False):
        super().__init__()
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_collate_fn = val_collate_fn if val_collate_fn is not None else collate_fn


        self.total_steps = None
        self.last_stepped_step = -1

        self.dl_workers = min(os.cpu_count()*2,8) if dl_workers==-1 else dl_workers

        self.train_ds = None
        self.val_ds = None

        self.train_dl = None
        self.val_dl = None

        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.ddp_sampler = ddp_sampler
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        if train_ds is not None:
            self.set_train_dataset(train_ds)

        if val_ds is not None:
            self.set_val_dataset(val_ds)

    def stats(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        freezed_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        total_params = trainable_params+freezed_params
        total_size = total_params * 4 / (1024 ** 2)
        print(tabulate([["Learnable",trainable_params],["Freezed",freezed_params],["Total",total_params]],headers=['Params','Count'],tablefmt='orgtbl', colalign=("left","right")))
        print(
            "\n",
            f"Model Size: {total_size:0.2f}MB",
            "\n",
        )
        
    def compile(self):
        torch_version = tuple(map(int, torch.__version__.split("+")[0].split('.')))
        if torch_version < (2, 0, 0):
            raise Exception("Model compilation is only supported for torch>=2.0.0")
        else:
            self.model = torch.compile(self.model, mode='reduce-overhead')
            print(f"Model is now compiled using torch.")
            
    def load_ckpt(self, checkpoint, ema=True, strict=True):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if ema and "ema_state_dict" in checkpoint:
            checkpoint_key = "ema_state_dict"
        else:
            if ema:
                print("Checkpoint does not contain 'ema_state_dict' key. Using 'state_dict' instead.")
            checkpoint_key = "state_dict"
        
        weights = checkpoint[checkpoint_key]
        first_key = list(weights.keys())[0]
        if first_key.split(".")[1] == "_orig_mod":
            print("Checkpoint file is from a compiled model. Compiling the model first...")
            self.compile()
        
        self.load_state_dict(weights, strict=strict)

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
                if self.train_sampler is None:
                    instance_sampler = DistributedSampler(self.train_ds)
                else:
                    instance_sampler = DistributedProxySampler(self.train_sampler)
            else:
                instance_sampler = self.train_sampler

            self.train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, sampler=instance_sampler, shuffle=True if instance_sampler is None else False, num_workers=self.dl_workers, collate_fn=self.collate_fn, pin_memory=self.pin_memory, drop_last=True, prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers)
            return self.train_dl

    def val_dataloader(self):
        if self.val_ds is None:
            raise Exception("Use the 'set_val_dataset' method to set the validation dataset.")
        else:
            if self.ddp_sampler:
                if self.val_sampler is None:
                    instance_sampler = DistributedSampler(self.val_ds)
                else:
                    instance_sampler = DistributedProxySampler(self.val_sampler)
            else:
                instance_sampler = self.val_sampler
                
            self.val_dl = torch.utils.data.DataLoader(self.val_ds, batch_size=self.val_batch_size, sampler=instance_sampler, shuffle=False, num_workers=self.dl_workers, collate_fn=self.val_collate_fn, pin_memory=self.pin_memory, drop_last=False, prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers)
            return self.val_dl

class EMA(pl.Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """
    def __init__(self, decay: float = 0.9999, ema_interval_steps: int = 1, ema_device: Optional[Union[torch.device, str]] = None, use_ema_for_validation: bool = True, pin_memory=True):
        super().__init__()
        self.decay = decay
        self.ema_interval_steps = ema_interval_steps
        self.ema_device: str = f"{ema_device}" if ema_device else "cuda:0"
        self.use_ema_for_validation = use_ema_for_validation
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False
        self.is_deepspeed_zero3 = None

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        return pl_module.state_dict()
        
    @overrides
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            strategy = trainer.strategy
            if isinstance(strategy, pl.pytorch.strategies.DeepSpeedStrategy):
                self.is_deepspeed_zero3 = strategy.zero_stage_3
            else:
                self.is_deepspeed_zero3 = False
                
            if self.is_deepspeed_zero:
                self.ema_state_dict = copy.deepcopy(self.get_state_dict(pl_module))
            else:
                self.ema_state_dict = copy.deepcopy(self.get_state_dict(pl_module))
                if self.ema_device:
                    self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in self.ema_state_dict.items()}

                if self.ema_device == "cpu" and self.ema_pin_memory:
                    self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

        self._ema_state_dict_ready = True

    @staticmethod
    def get_zero_param_status(param_list):
        return [
            p for p in param_list
            if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
        ]

    def moving_average(self, pl_module: pl.LightningModule, ema_state_dict: Dict[str, torch.Tensor], decay: float):
        with torch.no_grad():
            if self.is_deepspeed_zero3:
                for name, pl_param in pl_module.named_parameters():
                    if name in ema_state_dict:
                        ema_param = ema_state_dict[name]
                        params_to_fetch = EMA.get_zero_param_status([pl_param, ema_param])
                        should_gather_param = len(params_to_fetch) > 0
                        with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=should_gather_param):
                            print(ema_param.shape, pl_param.shape, name)
                            ema_state_dict[name].data.copy_(decay * ema_param.data + (1.0 - decay) * pl_param.data)
            else:
                for name, pl_param in pl_module.named_parameters():
                    if name in ema_state_dict:
                        ema_state_dict[name].data.copy_(decay * ema_state_dict[name].data + (1.0 - decay) * pl_param.data)

    @overrides
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, *args, **kwargs) -> None:
        if pl_module.global_rank == 0 and batch_idx % self.ema_interval_steps == 0:
            self.moving_average(pl_module, self.ema_state_dict, self.decay)

    @overrides  
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready or not self.use_ema_for_validation:
            return

        self.original_state_dict = copy.deepcopy(self.get_state_dict(pl_module))
        ema_state_dict = pl_module.trainer.strategy.broadcast(self.ema_state_dict, 0)
        self.ema_state_dict = ema_state_dict
        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
            f"Keys mismatch between EMA state dict and original state dict"
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            self.ema_state_dict = {}

    @overrides
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._ema_state_dict_ready or not self.use_ema_for_validation:
            return
        
        pl_module.load_state_dict(self.original_state_dict, strict=False)

    @overrides
    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        checkpoint["ema_state_dict"] = self.ema_state_dict
        checkpoint["_ema_state_dict_ready"] = self._ema_state_dict_ready

    @overrides
    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        self._ema_state_dict_ready = checkpoint["_ema_state_dict_ready"] 
        self.ema_state_dict = checkpoint["ema_state_dict"]
