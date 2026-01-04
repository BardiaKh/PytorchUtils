from typing import Optional, Union, Dict, Any
import os
import torch
import copy
from overrides import overrides
from tabulate import tabulate
import lightning as pl
from lightning.pytorch.utilities import rank_zero_only, grad_norm
from torch.utils.data.distributed import DistributedSampler
from .ddp_helper import DistributedProxySampler

class BKhModule(pl.LightningModule):
    def __init__(self, collate_fn=None, val_collate_fn=None, test_collate_fn=None, train_sampler=None, val_sampler=None, test_sampler=None, ddp_sampler=False, train_ds=None, val_ds=None, test_ds=None, dl_workers=-1, batch_size=None, val_batch_size=None, test_batch_size=None, pin_memory=True, prefetch_factor=1, persistent_workers=False):
        super().__init__()
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.test_batch_size = test_batch_size if test_batch_size is not None else self.val_batch_size
        self.val_collate_fn = val_collate_fn if val_collate_fn is not None else collate_fn
        self.test_collate_fn = test_collate_fn if test_collate_fn is not None else self.val_collate_fn

        self.total_steps = None
        self.last_stepped_step = -1

        self.dl_workers = min(os.cpu_count()*2,8) if dl_workers==-1 else dl_workers

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.train_dl = None
        self.val_dl = None
        self.test_dl = None

        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        self.ddp_sampler = ddp_sampler
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        if train_ds is not None:
            self.set_train_dataset(train_ds)

        if val_ds is not None:
            self.set_val_dataset(val_ds)

        if test_ds is not None:
            self.set_test_dataset(test_ds)

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
            checkpoint = torch.load(checkpoint, map_location=self.device)
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

    def set_test_dataset(self, ds):
        self.test_ds = ds

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

    def test_dataloader(self):
        if self.test_ds is None:
            raise Exception("Use the 'set_test_dataset' method to set the test dataset.")
        else:
            if self.ddp_sampler:
                if self.test_sampler is None:
                    instance_sampler = DistributedSampler(self.test_ds)
                else:
                    instance_sampler = DistributedProxySampler(self.test_sampler)
            else:
                instance_sampler = self.test_sampler
                
            self.test_dl = torch.utils.data.DataLoader(self.test_ds, batch_size=self.test_batch_size, sampler=instance_sampler, shuffle=False, num_workers=self.dl_workers, collate_fn=self.test_collate_fn, pin_memory=self.pin_memory, drop_last=False, prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers)
            return self.test_dl

class EMA(pl.Callback):
    def __init__(
        self,
        decay: float = 0.9999,
        ema_interval_steps: int = 1,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_warmup: bool = False,
        warmup_gamma: float = 1.0,
        warmup_power: float = 2/3,
        ema_device: Optional[Union[torch.device, str]] = None,  # Recommend None (same device)
        use_ema_for_validation: bool = True,
        foreach: bool = True,
        exclude_buffers: bool = True,  # Default True - buffers rarely need EMA
    ):
        super().__init__()
        self.decay = decay
        self.ema_interval_steps = ema_interval_steps
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.ema_device = ema_device
        self.use_ema_for_validation = use_ema_for_validation
        self.foreach = foreach
        self.exclude_buffers = exclude_buffers
        
        # Cached references - avoid repeated state_dict() calls
        self._ema_params: Optional[List[torch.Tensor]] = None
        self._model_params: Optional[List[torch.Tensor]] = None
        self._param_names: Optional[List[str]] = None
        self._ema_state_dict_ready = False

    def _init_ema_params(self, pl_module: pl.LightningModule):
        """Initialize EMA by cloning model parameters once."""
        self._param_names = []
        self._ema_params = []
        self._model_params = []
        
        for name, param in pl_module.named_parameters():
            self._param_names.append(name)
            self._model_params.append(param)
            # Clone to same device or specified device
            if self.ema_device:
                ema_p = param.data.clone().to(self.ema_device)
            else:
                ema_p = param.data.clone()
            self._ema_params.append(ema_p)
        
        # Optionally handle buffers
        if not self.exclude_buffers:
            for name, buf in pl_module.named_buffers():
                self._param_names.append(name)
                self._model_params.append(buf)
                if self.ema_device:
                    self._ema_params.append(buf.data.clone().to(self.ema_device))
                else:
                    self._ema_params.append(buf.data.clone())

    @overrides
    def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self._init_ema_params(pl_module)
        self._ema_state_dict_ready = True

    def get_decay(self, step: Optional[int] = None) -> float:
        if step is None:
            return self.decay
        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        if self.use_warmup:
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            return max(min(decay, self.decay), self.min_decay)
        return self.decay

    @rank_zero_only
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs) -> None:
        step = trainer.global_step
        if step % self.ema_interval_steps != 0:
            return
            
        decay = self.get_decay(step)
        if decay == 0.0:
            return
            
        self._update_ema(decay)

    @torch.no_grad()
    def _update_ema(self, decay: float):
        """Core EMA update - optimized to avoid allocations."""
        one_minus_decay = 1.0 - decay
        
        if self.foreach and self.ema_device is None:
            # Fastest path: same device, foreach ops
            # lerp_ is: ema = ema + (model - ema) * weight = ema * decay + model * (1-decay)
            torch._foreach_lerp_(self._ema_params, self._model_params, weight=one_minus_decay)
        elif self.foreach:
            # foreach but different devices - need to transfer
            model_on_device = [p.data.to(self.ema_device, non_blocking=True) for p in self._model_params]
            torch._foreach_lerp_(self._ema_params, model_on_device, weight=one_minus_decay)
        else:
            # Fallback: individual lerp
            for ema_p, model_p in zip(self._ema_params, self._model_params):
                if ema_p.is_floating_point():
                    ema_p.lerp_(model_p.data.to(ema_p.device, non_blocking=True), weight=one_minus_decay)
                else:
                    ema_p.copy_(model_p.data)

    def _get_ema_state_dict(self) -> Dict[str, torch.Tensor]:
        """Reconstruct state dict from cached params (only when needed)."""
        return dict(zip(self._param_names, self._ema_params))

    @overrides
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready or not self.use_ema_for_validation:
            return
        
        # Store current params (just the data tensors, not full clone)
        self._original_params = [p.data.clone() for p in self._model_params]
        
        # Broadcast and load EMA weights
        ema_state_dict = self._get_ema_state_dict()
        ema_state_dict = pl_module.trainer.strategy.broadcast(ema_state_dict, 0)
        pl_module.load_state_dict(ema_state_dict, strict=False)

    @overrides
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._ema_state_dict_ready or not self.use_ema_for_validation:
            return
        
        # Restore original params directly
        with torch.no_grad():
            for param, orig in zip(self._model_params, self._original_params):
                param.data.copy_(orig)
        self._original_params = None

    @overrides
    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        checkpoint["ema_state_dict"] = self._get_ema_state_dict()
        checkpoint["_ema_state_dict_ready"] = self._ema_state_dict_ready

    @overrides
    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        self._ema_state_dict_ready = checkpoint.get("_ema_state_dict_ready", False)
        ema_state_dict = checkpoint.get("ema_state_dict", {})
        if ema_state_dict:
            self._param_names = list(ema_state_dict.keys())
            self._ema_params = list(ema_state_dict.values())
            
class GradientNorm(pl.Callback):
    def __init__(self, norm_type: float = 2.0, log_on_step: bool = True, log_on_epoch: bool = False, log_on_progress_bar: bool = False):
        self.norm_type = norm_type
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.log_on_progress_bar = log_on_progress_bar
        super().__init__()

    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: "torch.optim.Optimizer") -> None:
        norms = grad_norm(pl_module, norm_type=self.norm_type)
        pl_module.log('grad_norm', norms[f'grad_{self.norm_type}_norm_total'], on_step=self.log_on_step, on_epoch=self.log_on_epoch, prog_bar=self.log_on_progress_bar, batch_size=1)
