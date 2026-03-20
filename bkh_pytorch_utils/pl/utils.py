from typing import Optional, Union, Dict, List, Any
import os
import torch
import copy
from overrides import overrides
from tabulate import tabulate
import lightning as pl
from lightning.pytorch.utilities import rank_zero_only, grad_norm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from .ddp_helper import DistributedProxySampler


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
DatasetInput = Union[Dataset, Dict[str, Dataset], None]
Broadcastable = Union[Any, Dict[str, Any], None]


# =============================================================================
# BKhModule
# =============================================================================

class BKhModule(pl.LightningModule):
    """
    Lightning module that transparently supports single or multi-dataset
    training.  When a dict of datasets is provided the module returns a
    dict of DataLoaders; Lightning will deliver a dict of batches to
    ``training_step`` / ``validation_step`` / ``test_step``.
    """

    def __init__(
        self,
        # --- collate ---
        collate_fn: Broadcastable = None,
        val_collate_fn: Broadcastable = None,
        test_collate_fn: Broadcastable = None,
        # --- samplers ---
        train_sampler: Broadcastable = None,
        val_sampler: Broadcastable = None,
        test_sampler: Broadcastable = None,
        ddp_sampler: bool = False,
        # --- datasets ---
        train_ds: DatasetInput = None,
        val_ds: DatasetInput = None,
        test_ds: DatasetInput = None,
        # --- loader settings ---
        dl_workers: Broadcastable = -1,
        batch_size: Broadcastable = None,
        val_batch_size: Broadcastable = None,
        test_batch_size: Broadcastable = None,
        pin_memory: Broadcastable = True,
        prefetch_factor: Broadcastable = 1,
        persistent_workers: Broadcastable = False,
    ):
        super().__init__()

        # ---- detect multi-dataset mode from whatever is passed first ----
        self._multi_ds_keys: Dict[str, set] = {
            "train": self._extract_keys(train_ds),
            "val":   self._extract_keys(val_ds),
            "test":  self._extract_keys(test_ds),
        }

        # ---- store collate fns (with fallback chain) ----
        self.collate_fn      = collate_fn
        self.val_collate_fn  = val_collate_fn if val_collate_fn is not None else collate_fn
        self.test_collate_fn = test_collate_fn if test_collate_fn is not None else self.val_collate_fn

        # ---- batch sizes (with fallback chain) ----
        self.batch_size      = batch_size
        self.val_batch_size  = val_batch_size if val_batch_size is not None else batch_size
        self.test_batch_size = test_batch_size if test_batch_size is not None else self.val_batch_size

        # ---- samplers ----
        self.train_sampler = train_sampler
        self.val_sampler   = val_sampler
        self.test_sampler  = test_sampler
        self.ddp_sampler   = ddp_sampler

        # ---- loader knobs ----
        self._dl_workers_raw  = dl_workers
        self.pin_memory       = pin_memory
        self.prefetch_factor  = prefetch_factor
        self.persistent_workers = persistent_workers

        # ---- bookkeeping ----
        self.total_steps       = None
        self.last_stepped_step = -1

        self.train_ds = None
        self.val_ds   = None
        self.test_ds  = None

        self.train_dl = None
        self.val_dl   = None
        self.test_dl  = None

        if train_ds is not None:
            self.set_train_dataset(train_ds)
        if val_ds is not None:
            self.set_val_dataset(val_ds)
        if test_ds is not None:
            self.set_test_dataset(test_ds)

    # =====================================================================
    # Helpers
    # =====================================================================

    @staticmethod
    def _extract_keys(ds) -> set:
        """Return the dict keys if *ds* is a dict, else empty set."""
        if isinstance(ds, dict):
            return set(ds.keys())
        return set()

    @staticmethod
    def _broadcast(value, keys: set):
        """
        If *value* is already a dict whose keys match *keys*, return it.
        Otherwise broadcast the scalar to every key.
        """
        if not keys:
            return value
        if isinstance(value, dict):
            missing = keys - set(value.keys())
            if missing:
                raise ValueError(
                    f"Dict parameter is missing keys: {missing}. "
                    f"Expected keys: {keys}"
                )
            return value
        return {k: value for k in keys}

    def _resolve_dl_workers(self, keys: set) -> Union[int, Dict[str, int]]:
        default = min(os.cpu_count() * 2, 8)
        raw = self._broadcast(self._dl_workers_raw, keys)
        if isinstance(raw, dict):
            return {k: (default if v == -1 else v) for k, v in raw.items()}
        return default if raw == -1 else raw

    # =====================================================================
    # Dataset setters (unchanged API — but now accept dicts too)
    # =====================================================================

    def set_train_dataset(self, ds):
        self._multi_ds_keys["train"] = self._extract_keys(ds)
        self.train_ds = ds

    def set_val_dataset(self, ds):
        self._multi_ds_keys["val"] = self._extract_keys(ds)
        self.val_ds = ds

    def set_test_dataset(self, ds):
        self._multi_ds_keys["test"] = self._extract_keys(ds)
        self.test_ds = ds

    # =====================================================================
    # Dataloader factories
    # =====================================================================

    def _build_dataloader(
        self,
        ds,
        batch_size,
        collate_fn,
        sampler,
        shuffle: bool,
        drop_last: bool,
        dl_workers,
        pin_memory,
        prefetch_factor,
        persistent_workers,
    ) -> DataLoader:
        """Build a single DataLoader."""
        if self.ddp_sampler:
            if sampler is None:
                instance_sampler = DistributedSampler(ds)
            else:
                instance_sampler = DistributedProxySampler(sampler)
        else:
            instance_sampler = sampler

        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=instance_sampler,
            shuffle=(shuffle if instance_sampler is None else False),
            num_workers=dl_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def _build_dataloaders(
        self,
        ds,
        batch_size,
        collate_fn,
        sampler,
        keys: set,
        shuffle: bool,
        drop_last: bool,
    ):
        """
        Return a single DataLoader *or* a dict of DataLoaders depending on
        whether the dataset is a dict.
        """
        dl_workers         = self._resolve_dl_workers(keys)
        pin_memory         = self._broadcast(self.pin_memory, keys)
        prefetch_factor    = self._broadcast(self.prefetch_factor, keys)
        persistent_workers = self._broadcast(self.persistent_workers, keys)

        if keys:  # multi-dataset
            bs_dict  = self._broadcast(batch_size, keys)
            cf_dict  = self._broadcast(collate_fn, keys)
            sm_dict  = self._broadcast(sampler, keys)
            dlw_dict = dl_workers if isinstance(dl_workers, dict) else {k: dl_workers for k in keys}
            pm_dict  = pin_memory if isinstance(pin_memory, dict) else {k: pin_memory for k in keys}
            pf_dict  = prefetch_factor if isinstance(prefetch_factor, dict) else {k: prefetch_factor for k in keys}
            pw_dict  = persistent_workers if isinstance(persistent_workers, dict) else {k: persistent_workers for k in keys}

            return {
                k: self._build_dataloader(
                    ds[k], bs_dict[k], cf_dict[k], sm_dict[k],
                    shuffle, drop_last,
                    dlw_dict[k], pm_dict[k], pf_dict[k], pw_dict[k],
                )
                for k in sorted(keys)
            }
        else:  # single dataset
            if isinstance(dl_workers, dict):
                dl_workers = next(iter(dl_workers.values()))
            return self._build_dataloader(
                ds, batch_size, collate_fn, sampler,
                shuffle, drop_last,
                dl_workers, pin_memory, prefetch_factor, persistent_workers,
            )

    # ---- public Lightning hooks ----

    def train_dataloader(self):
        if self.train_ds is None:
            raise RuntimeError("Use 'set_train_dataset' first.")
        keys = self._multi_ds_keys["train"]
        self.train_dl = self._build_dataloaders(
            self.train_ds, self.batch_size, self.collate_fn,
            self.train_sampler, keys, shuffle=True, drop_last=True,
        )
        return self.train_dl

    def val_dataloader(self):
        if self.val_ds is None:
            raise RuntimeError("Use 'set_val_dataset' first.")
        keys = self._multi_ds_keys["val"]
        self.val_dl = self._build_dataloaders(
            self.val_ds, self.val_batch_size, self.val_collate_fn,
            self.val_sampler, keys, shuffle=False, drop_last=False,
        )
        return self.val_dl

    def test_dataloader(self):
        if self.test_ds is None:
            raise RuntimeError("Use 'set_test_dataset' first.")
        keys = self._multi_ds_keys["test"]
        self.test_dl = self._build_dataloaders(
            self.test_ds, self.test_batch_size, self.test_collate_fn,
            self.test_sampler, keys, shuffle=False, drop_last=False,
        )
        return self.test_dl

    # =====================================================================
    # Utilities
    # =====================================================================

    def set_total_steps(self, steps=None, last_stepped_step=-1):
        if steps is not None:
            self.total_steps = steps
        if last_stepped_step is not None:
            self.last_stepped_step = last_stepped_step

    def forward(self, x):
        return self.model(x)

    def stats(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total     = trainable + frozen
        size_mb   = total * 4 / (1024 ** 2)
        print(tabulate(
            [["Learnable", trainable], ["Frozen", frozen], ["Total", total]],
            headers=["Params", "Count"], tablefmt="orgtbl", colalign=("left", "right"),
        ))
        print(f"\n Model Size: {size_mb:0.2f}MB\n")

    def compile(self):
        torch_version = tuple(map(int, torch.__version__.split("+")[0].split(".")))
        if torch_version < (2, 0, 0):
            raise RuntimeError("Compilation requires torch >= 2.0.0")
        self.model = torch.compile(self.model, mode="reduce-overhead")
        print("Model compiled with torch.compile.")

    def load_ckpt(self, checkpoint, ema=True, strict=True):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=self.device)
        key = "ema_state_dict" if (ema and "ema_state_dict" in checkpoint) else "state_dict"
        if ema and key == "state_dict":
            print("No 'ema_state_dict' found — falling back to 'state_dict'.")
        weights = checkpoint[key]
        if list(weights.keys())[0].split(".")[1] == "_orig_mod":
            print("Checkpoint from compiled model — compiling first...")
            self.compile()
        self.load_state_dict(weights, strict=strict)

    def get_best_checkpoint_path(self):
        if self.trainer is None:
            raise RuntimeError("Attach to a Trainer first via trainer.fit().")
        for cb in self.trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                p = cb.best_model_path
                if p.endswith(".ckpt"):
                    return p
        return None


# =============================================================================
# EMA Callback
# =============================================================================

class EMA(pl.Callback):
    """
    Exponential Moving Average of model parameters, implemented as a
    Lightning callback.

    Supports per-step updates, warmup schedules, cross-device storage,
    and automatic swapping for validation.
    """

    def __init__(
        self,
        decay: float = 0.9999,
        ema_interval_steps: int = 1,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_warmup: bool = False,
        warmup_gamma: float = 1.0,
        warmup_power: float = 2 / 3,
        ema_device: Optional[Union[torch.device, str]] = None,
        use_ema_for_validation: bool = True,
        foreach: bool = True,
        exclude_buffers: bool = True,
    ):
        super().__init__()

        self.decay              = decay
        self.ema_interval_steps = ema_interval_steps
        self.min_decay          = min_decay
        self.update_after_step  = update_after_step
        self.use_warmup         = use_warmup
        self.warmup_gamma       = warmup_gamma
        self.warmup_power       = warmup_power
        self.ema_device         = ema_device
        self.use_ema_for_validation = use_ema_for_validation
        self.foreach            = foreach
        self.exclude_buffers    = exclude_buffers

        # Cached references — avoid repeated state_dict() calls
        self._ema_params: Optional[List[torch.Tensor]]  = None
        self._model_params: Optional[List[torch.Tensor]] = None
        self._param_names: Optional[List[str]]           = None
        self._original_params: Optional[List[torch.Tensor]] = None
        self._ema_state_dict_ready = False

    # -----------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------

    def _init_ema_params(self, pl_module: pl.LightningModule):
        """Initialize EMA by cloning model parameters once."""
        self._param_names  = []
        self._ema_params   = []
        self._model_params = []

        for name, param in pl_module.named_parameters():
            self._param_names.append(name)
            self._model_params.append(param)
            if self.ema_device:
                self._ema_params.append(param.data.clone().to(self.ema_device))
            else:
                self._ema_params.append(param.data.clone())

        if not self.exclude_buffers:
            for name, buf in pl_module.named_buffers():
                self._param_names.append(name)
                self._model_params.append(buf)
                if self.ema_device:
                    self._ema_params.append(buf.data.clone().to(self.ema_device))
                else:
                    self._ema_params.append(buf.data.clone())

    # -----------------------------------------------------------------
    # Decay schedule
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Core EMA update
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _update_ema(self, decay: float):
        """Core EMA update — optimised to avoid allocations."""
        one_minus_decay = 1.0 - decay

        if self.foreach and self.ema_device is None:
            # Fastest path: same device, foreach ops
            torch._foreach_lerp_(self._ema_params, self._model_params, weight=one_minus_decay)
        elif self.foreach:
            # foreach but different devices — need to transfer
            model_on_device = [
                p.data.to(self.ema_device, non_blocking=True) for p in self._model_params
            ]
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

    # -----------------------------------------------------------------
    # Lightning hooks
    # -----------------------------------------------------------------

    @overrides
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self._init_ema_params(pl_module)
        self._ema_state_dict_ready = True

    @rank_zero_only
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs) -> None:
        step = trainer.global_step
        if step % self.ema_interval_steps != 0:
            return
        decay = self.get_decay(step)
        if decay == 0.0:
            return
        self._update_ema(decay)

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
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready or not self.use_ema_for_validation:
            return

        # Restore original params directly
        with torch.no_grad():
            for param, orig in zip(self._model_params, self._original_params):
                param.data.copy_(orig)
        self._original_params = None

    @overrides
    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Dict[str, Any]) -> None:
        checkpoint["ema_state_dict"] = self._get_ema_state_dict()
        checkpoint["_ema_state_dict_ready"] = self._ema_state_dict_ready

    @overrides
    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Dict[str, Any]) -> None:
        self._ema_state_dict_ready = checkpoint.get("_ema_state_dict_ready", False)
        ema_state_dict = checkpoint.get("ema_state_dict", {})
        if ema_state_dict:
            self._param_names = list(ema_state_dict.keys())
            self._ema_params  = list(ema_state_dict.values())


# =============================================================================
# Gradient Norm Callback
# =============================================================================

class GradientNorm(pl.Callback):
    """Efficient gradient norm logging using fused foreach norms (single GPU sync)."""
    def __init__(self, norm_type: float = 2.0):
        super().__init__()
        self.norm_type = norm_type

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        grads = [p.grad.detach() for p in pl_module.parameters() if p.grad is not None]
        if not grads:
            return
        # Use foreach_norm for fused kernel (1 launch instead of N)
        if hasattr(torch, '_foreach_norm') and self.norm_type == 2.0:
            per_norms = torch._foreach_norm(grads, self.norm_type)
            total_norm = torch.stack(per_norms).norm(self.norm_type)
        else:
            total_norm = torch.stack([g.norm(self.norm_type) for g in grads]).norm(self.norm_type)
        pl_module.log("grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=False, batch_size=1)
