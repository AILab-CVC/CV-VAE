import logging
import math
import re
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from packaging import version
from safetensors.torch import load_file as load_safetensors

from ..modules.autoencoding.regularizers import AbstractRegularizer
from ..modules.ema import LitEma
from ..util import (
    default,
    get_nested_attribute,
    get_obj_from_str,
    instantiate_from_config,
)

from ..modules.diffusionmodules.model_3d import Conv2dWithExtraDim

logpy = logging.getLogger(__name__)
# torch.autograd.set_detect_anomaly(True)


class AbstractAutoencoder(pl.LightningModule):
    """
    This is the base class for all autoencoders, including image autoencoders, image autoencoders with discriminators,
    unCLIP models, etc. Hence, it is fairly general, and specific features
    (e.g. discriminator training, encoding, decoding) must be implemented in subclasses.
    """

    def __init__(
        self,
        ema_decay: Union[None, float] = None,
        monitor: Union[None, str] = None,
        input_key: str = "jpg",
    ):
        super().__init__()

        self.input_key = input_key
        self.use_ema = ema_decay is not None
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)
            logpy.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            self.automatic_optimization = False

    # def apply_ckpt(self, ckpt: Union[None, str, dict]):
    #     if ckpt is None:
    #         return
    #     if isinstance(ckpt, str):
    #         ckpt = {
    #             "target": "lvdm.modules.checkpoint.CheckpointEngine",
    #             "params": {"ckpt_path": ckpt},
    #         }
    #     engine = instantiate_from_config(ckpt)
    #     engine(self)

    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        if ckpt is None:
            return
        if isinstance(ckpt, str):
            if ckpt.endswith("ckpt"):
                ckpt = torch.load(ckpt, map_location="cpu")["state_dict"]
            elif ckpt.endswith("safetensors"):
                ckpt = load_safetensors(ckpt)
            else:
                raise NotImplementedError

        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        print(
            f"Restored vae weights with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    @abstractmethod
    def get_input(self, batch) -> Any:
        raise NotImplementedError()

    def on_train_batch_end(self, *args, **kwargs):
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                logpy.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    logpy.info(f"{context}: Restored training weights")

    @abstractmethod
    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("encode()-method of abstract base class called")

    @abstractmethod
    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("decode()-method of abstract base class called")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        logpy.info(f"loading >>> {cfg['target']} <<< optimizer from config")
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def instantiate_scheduler_from_config(self, optimizer, cfg):
        logpy.info(
            f"loading >>> {cfg['target'] +' '+ cfg['params']['name']} <<< scheduler from config"
        )
        return get_obj_from_str(cfg["target"])(
            optimizer=optimizer, **cfg.get("params", dict())
        )

    def configure_optimizers(self) -> Any:
        raise NotImplementedError()


class AutoencodingEngine(AbstractAutoencoder):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        optimizer_config: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        lr_g_scheduler_config: Optional[dict] = None,
        lr_d_scheduler_config: Optional[dict] = None,
        trainable_ae_params: Optional[List[List[str]]] = None,
        ae_optimizer_args: Optional[List[dict]] = None,
        trainable_disc_params: Optional[List[List[str]]] = None,
        disc_optimizer_args: Optional[List[dict]] = None,
        disc_start_iter: int = 0,
        diff_boost_factor: float = 3.0,
        ckpt_engine: Union[None, str, dict] = None,
        ckpt_path: Optional[str] = None,
        additional_decode_keys: Optional[List[str]] = None,
        trainable: Union[bool, str] = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False  # pytorch lightning

        self.encoder: torch.nn.Module = instantiate_from_config(encoder_config)
        self.decoder: torch.nn.Module = instantiate_from_config(decoder_config)
        self.loss: torch.nn.Module = instantiate_from_config(loss_config)
        self.regularization: AbstractRegularizer = instantiate_from_config(
            regularizer_config
        )
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.Adam"}
        )
        self.lr_g_scheduler_config = default(
            lr_g_scheduler_config,
            {
                "target": "lvdm.lr_scheduler.get_scheduler",
                "params": {"name": "constant"},
            },
        )

        self.lr_d_scheduler_config = default(
            lr_d_scheduler_config,
            {
                "target": "lvdm.lr_scheduler.get_scheduler",
                "params": {"name": "constant"},
            },
        )
        self.diff_boost_factor = diff_boost_factor
        self.disc_start_iter = disc_start_iter
        self.lr_g_factor = lr_g_factor
        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [{} for _ in range(len(self.trainable_ae_params))],
            )
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [{}]  # makes type consitent

        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [{} for _ in range(len(self.trainable_disc_params))],
            )
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [{}]  # makes type consitent

        if ckpt_path is not None:
            assert ckpt_engine is None, "Can't set ckpt_engine and ckpt_path"
            logpy.warn("Checkpoint path is deprecated, use `checkpoint_egnine` instead")
        self.apply_ckpt(default(ckpt_path, ckpt_engine))
        self.additional_decode_keys = set(default(additional_decode_keys, []))
        assert trainable in [
            True,
            False,
            "decoder",
            "encoder",
        ], f"trainable ({trainable}) must in [True, False, decoder,encoder]"
        if not trainable:
            self.requires_grad_(False)
        elif trainable == "decoder":
            self.encoder.requires_grad_(False)
        elif trainable == "encoder":
            self.decoder.requires_grad_(False)

    def get_input(self, batch: Dict) -> torch.Tensor:
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in channels-first
        # format (e.g., bchw instead if bhwc)
        return batch[self.input_key]

    def get_autoencoder_params(self) -> list:
        params = []
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            params += list(self.loss.get_trainable_autoencoder_parameters())
        if hasattr(self.regularization, "get_trainable_parameters"):
            params += list(self.regularization.get_trainable_parameters())
        params = params + list(self.encoder.parameters())
        params = params + list(self.decoder.parameters())
        return params

    def get_discriminator_params(self) -> list:
        if hasattr(self.loss, "get_trainable_parameters"):
            params = list(self.loss.get_trainable_parameters())  # e.g., discriminator
        else:
            params = []
        return params

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x)
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.decoder(z, **kwargs)
        return x

    def forward(
        self, x: torch.Tensor, **additional_decode_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)
        dec = self.decode(z, **additional_decode_kwargs)
        return z, dec, reg_log

    def inner_training_step(
        self, batch: dict, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        x = self.get_input(batch)
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        z, xrec, regularization_log = self(x, **additional_decode_kwargs)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "train",
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        if optimizer_idx == 0:
            # autoencode
            out_loss = self.loss(x, xrec, **extra_info)
            if isinstance(out_loss, tuple):
                aeloss, log_dict_ae = out_loss
            else:
                # simple loss function
                aeloss = out_loss
                log_dict_ae = {"train/loss/rec": aeloss.detach()}

            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                "loss",
                aeloss.mean().detach(),
                prog_bar=True,
                logger=False,
                on_epoch=False,
                on_step=True,
            )
            return aeloss
        elif optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            # -> discriminator always needs to return a tuple
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return discloss
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.trainer.max_epochs - 1:
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            self.trainer.check_val_every_n_epoch = 1

            # Disable backward pass for SWA until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/17245)
            self.automatic_optimization = False

    def training_step(self, batch: dict, batch_idx: int):
        opts = self.optimizers()
        sche_g, sche_d = self.lr_schedulers()
        if not isinstance(opts, list):
            # Non-adversarial case
            opts = [opts]
        optimizer_idx = batch_idx % len(opts)
        if self.global_step < self.disc_start_iter:
            optimizer_idx = 0
        opt = opts[optimizer_idx]
        opt.zero_grad()
        with opt.toggle_model():
            loss = self.inner_training_step(
                batch, batch_idx, optimizer_idx=optimizer_idx
            )
            # with torch.autograd.detect_anomaly():
            self.manual_backward(loss)
            self.clip_gradients(
                opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )

        opt.step()
        sche_g.step()
        sche_d.step()

    def validation_step(self, batch: dict, batch_idx: int) -> Dict:
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
            log_dict.update(log_dict_ema)
        return log_dict

    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        x = self.get_input(batch)

        z, xrec, regularization_log = self(x)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": 0,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "val" + postfix,
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()
        out_loss = self.loss(x, xrec, **extra_info)
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss
        else:
            # simple loss function
            aeloss = out_loss
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}
        full_log_dict = log_dict_ae

        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            full_log_dict.update(log_dict_disc)
        self.log(
            f"val{postfix}/loss/rec",
            log_dict_ae[f"val{postfix}/loss/rec"],
            sync_dist=True,
        )
        self.log_dict(full_log_dict, sync_dist=True)
        return full_log_dict

    def get_param_groups(
        self, parameter_names: List[List[str]], optimizer_args: List[dict]
    ) -> Tuple[List[Dict[str, Any]], int]:
        groups = []
        num_params = 0
        for names, args in zip(parameter_names, optimizer_args):
            params = []
            for pattern_ in names:
                pattern_params = []
                pattern = re.compile(pattern_)
                for p_name, param in self.named_parameters():
                    if re.match(pattern, p_name):
                        pattern_params.append(param)
                        num_params += param.numel()
                if len(pattern_params) == 0:
                    logpy.warn(f"Did not find parameters for pattern {pattern_}")
                params.extend(pattern_params)
            groups.append({"params": params, **args})
        return groups, num_params

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        if self.trainable_ae_params is None:
            ae_params = self.get_autoencoder_params()
        else:
            ae_params, num_ae_params = self.get_param_groups(
                self.trainable_ae_params, self.ae_optimizer_args
            )
            logpy.info(f"Number of trainable autoencoder parameters: {num_ae_params:,}")
        if self.trainable_disc_params is None:
            disc_params = self.get_discriminator_params()
        else:
            disc_params, num_disc_params = self.get_param_groups(
                self.trainable_disc_params, self.disc_optimizer_args
            )
            logpy.info(
                f"Number of trainable discriminator parameters: {num_disc_params:,}"
            )
        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        sche_ae = self.instantiate_scheduler_from_config(
            opt_ae, self.lr_g_scheduler_config
        )
        opts = [opt_ae]
        sches = [sche_ae]
        if len(disc_params) > 0:
            opt_disc = self.instantiate_optimizer_from_config(
                disc_params, self.learning_rate, self.optimizer_config
            )
            opts.append(opt_disc)
            sche_disc = self.instantiate_scheduler_from_config(
                opt_disc, self.lr_d_scheduler_config
            )
            sches.append(sche_disc)

        return opts, sches

    @torch.no_grad()
    def log_images(
        self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs
    ) -> dict:
        log = dict()
        additional_decode_kwargs = {}
        x = self.get_input(batch)
        additional_decode_kwargs.update(
            {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
        )

        _, xrec, _ = self(x, **additional_decode_kwargs)
        is_video = x.dim() == 5
        if is_video:
            b, c, t, h, w = x.shape
            x = rearrange(x, "b c t h w -> (b t) c h w")
            xrec = rearrange(xrec, "b c t h w -> (b t) c h w")

        log["inputs"] = x
        log["reconstructions"] = xrec
        diff = 0.5 * torch.abs(torch.clamp(xrec, -1.0, 1.0) - x)
        diff.clamp_(0, 1.0)
        log["diff"] = 2.0 * diff - 1.0
        # diff_boost shows location of small errors, by boosting their
        # brightness.
        log["diff_boost"] = (
            2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1
        )
        if is_video:
            x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
            xrec = rearrange(xrec, "(b t) c h w -> b c t h w", b=b, t=t)
        # if hasattr(self.loss, "log_images"):
        #     log.update(self.loss.log_images(x, xrec))
        with self.ema_scope():
            _, xrec_ema, _ = self(x, **additional_decode_kwargs)

        if is_video:
            x = rearrange(x, "b c t h w -> (b t) c h w")
            xrec_ema = rearrange(xrec_ema, "b c t h w -> (b t) c h w")

            log["reconstructions_ema"] = xrec_ema
            diff_ema = 0.5 * torch.abs(torch.clamp(xrec_ema, -1.0, 1.0) - x)
            diff_ema.clamp_(0, 1.0)
            log["diff_ema"] = 2.0 * diff_ema - 1.0
            log["diff_boost_ema"] = (
                2.0 * torch.clamp(self.diff_boost_factor * diff_ema, 0.0, 1.0) - 1
            )
        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            if is_video:
                x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
            _, xrec_add, _ = self(x, **additional_decode_kwargs)
            if is_video:
                x = rearrange(x, "b c t h w -> (b t) c h w")
                xrec_add = rearrange(xrec_add, "b c t h w -> (b t) c h w")
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            log[log_str] = xrec_add
        return log


class AutoencodingEngineLegacy(AutoencodingEngine):
    def __init__(self, embed_dim: int, **kwargs):
        self.max_batch_size = kwargs.pop("max_batch_size", None)
        ddconfig = kwargs.pop("ddconfig")
        ckpt_path = kwargs.pop("ckpt_path", None)
        ckpt_engine = kwargs.pop("ckpt_engine", None)
        super().__init__(
            encoder_config={
                "target": "lvdm.modules.diffusionmodules.model.Encoder",
                "params": ddconfig,
            },
            decoder_config={
                "target": "lvdm.modules.diffusionmodules.model.Decoder",
                "params": ddconfig,
            },
            **kwargs,
        )
        self.quant_conv = Conv2dWithExtraDim(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1,
        )
        self.post_quant_conv = Conv2dWithExtraDim(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        self.apply_ckpt(default(ckpt_path, ckpt_engine))

    def get_autoencoder_params(self) -> list:
        params = super().get_autoencoder_params()
        return params

    def encode(
        self, x: torch.Tensor, return_reg_log: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        if self.max_batch_size is None:
            z = self.encoder(x)
            z = self.quant_conv(z)
        else:
            N = x.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            z = list()
            for i_batch in range(n_batches):
                z_batch = self.encoder(x[i_batch * bs : (i_batch + 1) * bs])
                z_batch = self.quant_conv(z_batch)
                z.append(z_batch)
            z = torch.cat(z, 0)

        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        if self.max_batch_size is None:
            dec = self.post_quant_conv(z)
            dec = self.decoder(dec, **decoder_kwargs)
        else:
            N = z.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            dec = list()
            for i_batch in range(n_batches):
                dec_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                dec_batch = self.decoder(dec_batch, **decoder_kwargs)
                dec.append(dec_batch)
            dec = torch.cat(dec, 0)

        return dec


class AutoencoderKL(AutoencodingEngineLegacy):
    def __init__(self, **kwargs):
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")
        super().__init__(
            regularizer_config={
                "target": (
                    "lvdm.modules.autoencoding.regularizers"
                    ".DiagonalGaussianRegularizer"
                )
            },
            **kwargs,
        )


class AutoencoderLegacyVQ(AutoencodingEngineLegacy):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        sane_index_shape: bool = False,
        **kwargs,
    ):
        if "lossconfig" in kwargs:
            logpy.warn(f"Parameter `lossconfig` is deprecated, use `loss_config`.")
            kwargs["loss_config"] = kwargs.pop("lossconfig")
        super().__init__(
            regularizer_config={
                "target": (
                    "lvdm.modules.autoencoding.regularizers.quantize" ".VectorQuantizer"
                ),
                "params": {
                    "n_e": n_embed,
                    "e_dim": embed_dim,
                    "sane_index_shape": sane_index_shape,
                },
            },
            **kwargs,
        )


class IdentityFirstStage(AbstractAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_input(self, x: Any) -> Any:
        return x

    def encode(self, x: Any, *args, **kwargs) -> Any:
        return x

    def decode(self, x: Any, *args, **kwargs) -> Any:
        return x


class AEIntegerWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        shape: Union[None, Tuple[int, int], List[int]] = (16, 16),
        regularization_key: str = "regularization",
        encoder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model
        assert hasattr(model, "encode") and hasattr(
            model, "decode"
        ), "Need AE interface"
        self.regularization = get_nested_attribute(model, regularization_key)
        self.shape = shape
        self.encoder_kwargs = default(encoder_kwargs, {"return_reg_log": True})

    def encode(self, x) -> torch.Tensor:
        assert (
            not self.training
        ), f"{self.__class__.__name__} only supports inference currently"
        _, log = self.model.encode(x, **self.encoder_kwargs)
        assert isinstance(log, dict)
        inds = log["min_encoding_indices"]
        return rearrange(inds, "b ... -> b (...)")

    def decode(
        self, inds: torch.Tensor, shape: Union[None, tuple, list] = None
    ) -> torch.Tensor:
        # expect inds shape (b, s) with s = h*w
        shape = default(shape, self.shape)  # Optional[(h, w)]
        if shape is not None:
            assert len(shape) == 2, f"Unhandeled shape {shape}"
            inds = rearrange(inds, "b (h w) -> b h w", h=shape[0], w=shape[1])
        h = self.regularization.get_codebook_entry(inds)  # (b, h, w, c)
        h = rearrange(h, "b h w c -> b c h w")
        return self.model.decode(h)


class AutoencoderKLModeOnly(AutoencodingEngineLegacy):
    def __init__(self, **kwargs):
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")
        super().__init__(
            regularizer_config={
                "target": (
                    "lvdm.modules.autoencoding.regularizers"
                    ".DiagonalGaussianRegularizer"
                ),
                "params": {"sample": False},
            },
            **kwargs,
        )


class Autoencoding3DEngine(AutoencodingEngine):
    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        optimizer_config: Dict | None = None,
        lr_g_factor: float = 1,
        lr_g_scheduler_config: Dict | None = None,
        lr_d_scheduler_config: Dict | None = None,
        trainable_ae_params: List[List[str]] | None = None,
        ae_optimizer_args: List[Dict] | None = None,
        trainable_disc_params: List[List[str]] | None = None,
        disc_optimizer_args: List[Dict] | None = None,
        disc_start_iter: int = 0,
        diff_boost_factor: float = 3,
        ckpt_engine: None | str | Dict = None,
        ckpt_path: str | None = None,
        additional_decode_keys: List[str] | None = None,
        en_de_n_frames_a_time: Optional[int] = None,
        time_n_compress: Optional[int] = None,
        spatial_n_compress: Optional[int] = None,
        tile_spatial_size: Optional[int] = None,
        num_video_frames: Optional[int] = None,
        tile_overlap_ratio: Optional[float] = None,
        reshape_z_dim_to_4: bool = False,
        reshape_x_dim_to_4: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            loss_config=loss_config,
            regularizer_config=regularizer_config,
            optimizer_config=optimizer_config,
            lr_g_factor=lr_g_factor,
            lr_g_scheduler_config=lr_g_scheduler_config,
            lr_d_scheduler_config=lr_d_scheduler_config,
            trainable_ae_params=trainable_ae_params,
            ae_optimizer_args=ae_optimizer_args,
            trainable_disc_params=trainable_disc_params,
            disc_optimizer_args=disc_optimizer_args,
            disc_start_iter=disc_start_iter,
            diff_boost_factor=diff_boost_factor,
            ckpt_engine=ckpt_engine,
            ckpt_path=ckpt_path,
            additional_decode_keys=additional_decode_keys,
            **kwargs,
        )
        if en_de_n_frames_a_time is not None:
            assert time_n_compress is not None
            assert en_de_n_frames_a_time % time_n_compress == 0
            self.encode_n_frames_a_time = en_de_n_frames_a_time
            self.decode_n_frames_a_time = en_de_n_frames_a_time // time_n_compress
        else:
            self.encode_n_frames_a_time = None
            self.decode_n_frames_a_time = None

        if num_video_frames is not None:
            assert time_n_compress is not None
            self.num_video_frames = num_video_frames
            self.num_latent_frames = 1 + (num_video_frames - 1) // time_n_compress
        else:
            self.num_video_frames = None
            self.num_latent_frames = None

        if tile_spatial_size is not None:
            assert spatial_n_compress is not None and tile_overlap_ratio is not None
            self.pixel_tile_size = tile_spatial_size
            self.latent_tile_size = tile_spatial_size // spatial_n_compress
            self.tile_overlap_ratio = tile_overlap_ratio
        else:
            self.pixel_tile_size = None
            self.latent_tile_size = None
            self.tile_overlap_ratio = None

        self.reshape_z_dim_to_4 = reshape_z_dim_to_4
        self.reshape_x_dim_to_4 = reshape_x_dim_to_4

    def spatial_tiled_encode(self, x):
        if self.pixel_tile_size is None:
            z = self.encoder(x)
        else:
            pixel_stride = round(self.pixel_tile_size * (1 - self.tile_overlap_ratio))
            latent_overlap = round(self.latent_tile_size * self.tile_overlap_ratio)
            latent_stride = self.latent_tile_size - latent_overlap
            rows = []
            for i in range(0, x.shape[3], pixel_stride):
                cols = []
                for j in range(0, x.shape[4], pixel_stride):
                    tile = x[
                        :,
                        :,
                        :,
                        i : i + self.pixel_tile_size,
                        j : j + self.pixel_tile_size,
                    ]
                    tile = self.encoder(tile)
                    cols.append(tile)
                    if j + self.pixel_tile_size >= x.shape[4]:
                        break
                rows.append(cols)
                if i + self.pixel_tile_size >= x.shape[3]:
                    break

            results_rows = []
            for i, cols in enumerate(rows):
                results_cols = []
                for j, tile in enumerate(cols):
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, latent_overlap)
                    if j > 0:
                        tile = self.blend_h(cols[j - 1], tile, latent_overlap)
                    results_cols.append(tile)
                results_rows.append(results_cols)

            latents = []
            for i, cols in enumerate(results_rows):
                for j, tile in enumerate(cols):
                    if i < len(results_rows) - 1:
                        tile = tile[:, :, :, :latent_stride, :]
                    if j < len(cols) - 1:
                        tile = tile[:, :, :, :, :latent_stride]
                    cols[j] = tile
                latents.append(torch.cat(cols, dim=4))
            z = torch.cat(latents, dim=3)
        return z

    def tiled_encode(self, x):
        if self.encode_n_frames_a_time is None:
            z = self.spatial_tiled_encode(x)
        else:
            assert x.dim() == 5
            z_all = []
            stride = self.encode_n_frames_a_time
            n_rounds = math.ceil((x.shape[2] - 1) / stride)
            n_rounds = 1 if n_rounds == 0 else n_rounds
            for n in range(n_rounds):
                z_i = self.spatial_tiled_encode(
                    x[:, :, n * stride : (n + 1) * stride + 1, :, :]
                )
                z_i = z_i if n == 0 else z_i[:, :, 1:, :, :]
                z_all.append(z_i)
            z = torch.cat(z_all, dim=2)

        return z

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        if x.dim() == 4:
            if self.num_video_frames is not None:
                x = rearrange(x, "(b t) c h w -> b c t h w", t=self.num_video_frames)
            else:
                x = rearrange(x, "b c h w -> b c () h w")

        z = self.tiled_encode(x)

        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        if self.reshape_z_dim_to_4:
            z = rearrange(z, "b c t h w -> (b t) c h w")
        return z

    def spatial_tiled_decode(self, z: torch.Tensor, **kwargs):
        if self.latent_tile_size is None:
            x = self.decoder(z, **kwargs)
        else:
            latent_stride = round(self.latent_tile_size * (1 - self.tile_overlap_ratio))
            pixel_overlap = round(self.pixel_tile_size * self.tile_overlap_ratio)
            pixel_stride = self.pixel_tile_size - pixel_overlap

            rows = []
            for i in range(0, z.shape[3], latent_stride):
                cols = []
                for j in range(0, z.shape[4], latent_stride):
                    tile = z[
                        :,
                        :,
                        :,
                        i : i + self.latent_tile_size,
                        j : j + self.latent_tile_size,
                    ]
                    tile = self.decoder(tile)
                    cols.append(tile)
                    if j + self.latent_tile_size >= z.shape[4]:
                        break
                rows.append(cols)
                if i + self.latent_tile_size >= z.shape[3]:
                    break
            results_rows = []
            for i, cols in enumerate(rows):
                results_cols = []
                for j, tile in enumerate(cols):
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, pixel_overlap)
                    if j > 0:
                        tile = self.blend_h(cols[j - 1], tile, pixel_overlap)
                    results_cols.append(tile)
                results_rows.append(results_cols)

            pixels = []
            for i, cols in enumerate(results_rows):
                for j, tile in enumerate(cols):
                    if i < len(results_rows) - 1:
                        tile = tile[:, :, :, :pixel_stride, :]
                    if j < len(cols) - 1:
                        tile = tile[:, :, :, :, :pixel_stride]
                    cols[j] = tile
                pixels.append(torch.cat(cols, dim=4))
            x = torch.cat(pixels, dim=3)
        return x

    def tiled_decode(self, z: torch.Tensor, **kwargs):
        if self.decode_n_frames_a_time is None:
            x = self.spatial_tiled_decode(z, **kwargs)
        else:
            assert z.dim() == 5
            x_all = []
            stride = self.decode_n_frames_a_time
            n_rounds = math.ceil((z.shape[2] - 1) / stride)
            n_rounds = 1 if n_rounds == 0 else n_rounds

            for n in range(n_rounds):
                x_i = self.spatial_tiled_decode(
                    z[:, :, n * stride : (n + 1) * stride + 1, :, :], **kwargs
                )
                x_i = x_i if n == 0 else x_i[:, :, 1:, :, :]
                x_all.append(x_i)
            x = torch.cat(x_all, dim=2)
        return x

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        if z.dim() == 4:
            if self.num_latent_frames is not None:
                z = rearrange(z, "(b t) c h w -> b c t h w", t=self.num_latent_frames)
            else:
                z = rearrange(z, "b c h w -> b c () h w")
        x = self.tiled_decode(z, **kwargs)
        if self.reshape_x_dim_to_4:
            x = rearrange(x, "b c t h w -> (b t) c h w")
        return x

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, overlap_size: int
    ) -> torch.Tensor:
        weight_b = (torch.arange(overlap_size).view(1, 1, 1, 1, -1) / overlap_size).to(
            b.device
        )
        b[:, :, :, :, :overlap_size] = (1 - weight_b) * a[
            :, :, :, :, -overlap_size:
        ] + weight_b * b[:, :, :, :, :overlap_size]
        return b

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, overlap_size: int
    ) -> torch.Tensor:
        weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1, 1) / overlap_size).to(
            b.device
        )
        b[:, :, :, :overlap_size, :] = (1 - weight_b) * a[
            :, :, :, -overlap_size:, :
        ] + weight_b * b[:, :, :, :overlap_size, :]
        return b


class AutoencodingEngineWithLatentConstraint(AutoencodingEngine):
    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        constraint_decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        optimizer_config: Dict | None = None,
        lr_g_factor: float = 1,
        lr_g_scheduler_config: Dict | None = None,
        lr_d_scheduler_config: Dict | None = None,
        trainable_ae_params: List[List[str]] | None = None,
        ae_optimizer_args: List[Dict] | None = None,
        trainable_disc_params: List[List[str]] | None = None,
        disc_optimizer_args: List[Dict] | None = None,
        disc_start_iter: int = 0,
        diff_boost_factor: float = 3,
        ckpt_engine: None | str | Dict = None,
        ckpt_path: str | None = None,
        additional_decode_keys: List[str] | None = None,
        trainable: bool | str = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            loss_config=loss_config,
            regularizer_config=regularizer_config,
            optimizer_config=optimizer_config,
            lr_g_factor=lr_g_factor,
            lr_g_scheduler_config=lr_g_scheduler_config,
            lr_d_scheduler_config=lr_d_scheduler_config,
            trainable_ae_params=trainable_ae_params,
            ae_optimizer_args=ae_optimizer_args,
            trainable_disc_params=trainable_disc_params,
            disc_optimizer_args=disc_optimizer_args,
            disc_start_iter=disc_start_iter,
            diff_boost_factor=diff_boost_factor,
            ckpt_engine=None,
            ckpt_path=None,
            additional_decode_keys=additional_decode_keys,
            trainable=trainable,
            **kwargs,
        )

        self.constraint_decoder = instantiate_from_config(constraint_decoder_config)
        self.constraint_decoder.requires_grad_(False)
        if ckpt_path is not None:
            assert ckpt_engine is None, "Can't set ckpt_engine and ckpt_path"
            logpy.warn("Checkpoint path is deprecated, use `checkpoint_egnine` instead")
        self.apply_ckpt(default(ckpt_path, ckpt_engine))

    def inner_training_step(
        self, batch: dict, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        x = self.get_input(batch)
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        z, xrec, regularization_log = self(x, **additional_decode_kwargs)
        xrec_2d = self.constraint_decoder(z, **additional_decode_kwargs)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "train",
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        if optimizer_idx == 0:
            # autoencode
            out_loss = self.loss(x, xrec, xrec_2d, **extra_info)
            if isinstance(out_loss, tuple):
                aeloss, log_dict_ae = out_loss
            else:
                # simple loss function
                aeloss = out_loss
                log_dict_ae = {"train/loss/rec": aeloss.detach()}

            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                "loss",
                aeloss.mean().detach(),
                prog_bar=True,
                logger=False,
                on_epoch=False,
                on_step=True,
            )
            return aeloss
        elif optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, xrec_2d, **extra_info)
            # -> discriminator always needs to return a tuple
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return discloss
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")

    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        x = self.get_input(batch)

        z, xrec, regularization_log = self(x)
        xrec_2d = self.constraint_decoder(z)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": 0,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "val" + postfix,
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()
        out_loss = self.loss(x, xrec, xrec_2d, **extra_info)
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss
        else:
            # simple loss function
            aeloss = out_loss
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}
        full_log_dict = log_dict_ae

        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1
            discloss, log_dict_disc = self.loss(x, xrec, xrec_2d, **extra_info)
            full_log_dict.update(log_dict_disc)
        self.log(
            f"val{postfix}/loss/rec",
            log_dict_ae[f"val{postfix}/loss/rec"],
            sync_dist=True,
        )
        self.log_dict(full_log_dict, sync_dist=True)
        return full_log_dict

    @torch.no_grad()
    def log_images(
        self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs
    ) -> dict:
        log = dict()
        additional_decode_kwargs = {}
        x = self.get_input(batch)
        additional_decode_kwargs.update(
            {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
        )

        z, xrec, _ = self(x, **additional_decode_kwargs)
        xrec_2d = self.constraint_decoder(z, **additional_decode_kwargs)
        is_video = x.dim() == 5
        if is_video:
            b, c, t, h, w = x.shape
            x = rearrange(x, "b c t h w -> (b t) c h w")
            xrec = rearrange(xrec, "b c t h w -> (b t) c h w")
            xrec_2d = rearrange(xrec_2d, "b c t h w -> (b t) c h w")

        log["inputs"] = x
        log["reconstructions"] = xrec
        log["reconstructions_2d"] = xrec_2d
        diff = 0.5 * torch.abs(torch.clamp(xrec, -1.0, 1.0) - x)
        diff.clamp_(0, 1.0)
        log["diff"] = 2.0 * diff - 1.0
        # diff_boost shows location of small errors, by boosting their
        # brightness.
        log["diff_boost"] = (
            2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1
        )
        if is_video:
            x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
            xrec = rearrange(xrec, "(b t) c h w -> b c t h w", b=b, t=t)
        # if hasattr(self.loss, "log_images"):
        #     log.update(self.loss.log_images(x, xrec))
        with self.ema_scope():
            _, xrec_ema, _ = self(x, **additional_decode_kwargs)

        if is_video:
            x = rearrange(x, "b c t h w -> (b t) c h w")
            xrec_ema = rearrange(xrec_ema, "b c t h w -> (b t) c h w")

            log["reconstructions_ema"] = xrec_ema
            diff_ema = 0.5 * torch.abs(torch.clamp(xrec_ema, -1.0, 1.0) - x)
            diff_ema.clamp_(0, 1.0)
            log["diff_ema"] = 2.0 * diff_ema - 1.0
            log["diff_boost_ema"] = (
                2.0 * torch.clamp(self.diff_boost_factor * diff_ema, 0.0, 1.0) - 1
            )
        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            if is_video:
                x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
            _, xrec_add, _ = self(x, **additional_decode_kwargs)
            if is_video:
                x = rearrange(x, "b c t h w -> (b t) c h w")
                xrec_add = rearrange(xrec_add, "b c t h w -> (b t) c h w")
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            log[log_str] = xrec_add
        return log


class AutoencodingEngineWithEncoderConstraint(AutoencodingEngine):
    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        constraint_encoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        optimizer_config: Dict | None = None,
        lr_g_factor: float = 1,
        lr_g_scheduler_config: Dict | None = None,
        lr_d_scheduler_config: Dict | None = None,
        trainable_ae_params: List[List[str]] | None = None,
        ae_optimizer_args: List[Dict] | None = None,
        trainable_disc_params: List[List[str]] | None = None,
        disc_optimizer_args: List[Dict] | None = None,
        disc_start_iter: int = 0,
        diff_boost_factor: float = 3,
        ckpt_engine: None | str | Dict = None,
        ckpt_path: str | None = None,
        additional_decode_keys: List[str] | None = None,
        trainable: bool | str = True,
        time_n_compress: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            loss_config=loss_config,
            regularizer_config=regularizer_config,
            optimizer_config=optimizer_config,
            lr_g_factor=lr_g_factor,
            lr_g_scheduler_config=lr_g_scheduler_config,
            lr_d_scheduler_config=lr_d_scheduler_config,
            trainable_ae_params=trainable_ae_params,
            ae_optimizer_args=ae_optimizer_args,
            trainable_disc_params=trainable_disc_params,
            disc_optimizer_args=disc_optimizer_args,
            disc_start_iter=disc_start_iter,
            diff_boost_factor=diff_boost_factor,
            ckpt_engine=None,
            ckpt_path=None,
            additional_decode_keys=additional_decode_keys,
            trainable=trainable,
            **kwargs,
        )

        self.constraint_encoder = instantiate_from_config(constraint_encoder_config)
        self.constraint_encoder.requires_grad_(False)
        if ckpt_path is not None:
            assert ckpt_engine is None, "Can't set ckpt_engine and ckpt_path"
            logpy.warn("Checkpoint path is deprecated, use `checkpoint_egnine` instead")
        self.apply_ckpt(default(ckpt_path, ckpt_engine))
        self.time_n_compress = time_n_compress

    def forward(
        self, x: torch.Tensor, **additional_decode_kwargs
    ) -> Tuple[torch.Tensor | Dict]:
        x_d = x[:, :, :: self.time_n_compress, :, :]
        with torch.no_grad():
            z_d = self.constraint_encoder(x_d)
        z = self.encoder(x)
        z = torch.cat([z, z_d], dim=0)
        z, reg_log = self.regularization(z)
        dec = self.decoder(z, **additional_decode_kwargs)
        return z, dec, reg_log

    def inner_training_step(
        self, batch: dict, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        x = self.get_input(batch)
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        z, xrec, regularization_log = self(x, **additional_decode_kwargs)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "train",
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        if optimizer_idx == 0:
            # autoencode
            out_loss = self.loss(x, xrec, **extra_info)
            if isinstance(out_loss, tuple):
                aeloss, log_dict_ae = out_loss
            else:
                # simple loss function
                aeloss = out_loss
                log_dict_ae = {"train/loss/rec": aeloss.detach()}

            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                "loss",
                aeloss.mean().detach(),
                prog_bar=True,
                logger=False,
                on_epoch=False,
                on_step=True,
            )
            return aeloss
        elif optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            # -> discriminator always needs to return a tuple
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return discloss
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")

    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        x = self.get_input(batch)

        z, xrec, regularization_log = self(x)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": 0,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "val" + postfix,
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()
        out_loss = self.loss(x, xrec, **extra_info)
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss
        else:
            # simple loss function
            aeloss = out_loss
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}
        full_log_dict = log_dict_ae

        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            full_log_dict.update(log_dict_disc)
        self.log(
            f"val{postfix}/loss/rec",
            log_dict_ae[f"val{postfix}/loss/rec"],
            sync_dist=True,
        )
        self.log_dict(full_log_dict, sync_dist=True)
        return full_log_dict

    @torch.no_grad()
    def log_images(
        self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs
    ) -> dict:
        log = dict()
        additional_decode_kwargs = {}
        x = self.get_input(batch)
        additional_decode_kwargs.update(
            {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
        )

        z, xrec, _ = self(x, **additional_decode_kwargs)
        xrec, xrec_d = torch.chunk(xrec, 2)
        is_video = x.dim() == 5
        if is_video:
            b, c, t, h, w = x.shape
            x = rearrange(x, "b c t h w -> (b t) c h w")
            xrec = rearrange(xrec, "b c t h w -> (b t) c h w")
            xrec_d = rearrange(xrec_d, "b c t h w -> (b t) c h w")

        log["inputs"] = x
        log["reconstructions"] = xrec
        log["rec_d"] = xrec_d

        diff = 0.5 * torch.abs(torch.clamp(xrec, -1.0, 1.0) - x)
        diff.clamp_(0, 1.0)
        log["diff"] = 2.0 * diff - 1.0
        # diff_boost shows location of small errors, by boosting their
        # brightness.
        log["diff_boost"] = (
            2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1
        )
        if is_video:
            x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
            xrec = rearrange(xrec, "(b t) c h w -> b c t h w", b=b, t=t)
        # if hasattr(self.loss, "log_images"):
        #     log.update(self.loss.log_images(x, xrec))
        with self.ema_scope():
            _, xrec_ema, _ = self(x, **additional_decode_kwargs)
            xrec_ema, _ = torch.chunk(xrec_ema, 2)

        if is_video:
            x = rearrange(x, "b c t h w -> (b t) c h w")
            xrec_ema = rearrange(xrec_ema, "b c t h w -> (b t) c h w")

            log["reconstructions_ema"] = xrec_ema
            diff_ema = 0.5 * torch.abs(torch.clamp(xrec_ema, -1.0, 1.0) - x)
            diff_ema.clamp_(0, 1.0)
            log["diff_ema"] = 2.0 * diff_ema - 1.0
            log["diff_boost_ema"] = (
                2.0 * torch.clamp(self.diff_boost_factor * diff_ema, 0.0, 1.0) - 1
            )
        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            if is_video:
                x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
            _, xrec_add, _ = self(x, **additional_decode_kwargs)
            xrec_add, _ = torch.chunk(xrec_add, 2)
            if is_video:
                x = rearrange(x, "b c t h w -> (b t) c h w")
                xrec_add = rearrange(xrec_add, "b c t h w -> (b t) c h w")
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            log[log_str] = xrec_add
        return log


class AutoencodingEngineWithAllConstraint(AutoencodingEngine):
    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        constraint_encoder_config: Dict,
        constraint_decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        optimizer_config: Dict | None = None,
        lr_g_factor: float = 1,
        lr_g_scheduler_config: Dict | None = None,
        lr_d_scheduler_config: Dict | None = None,
        trainable_ae_params: List[List[str]] | None = None,
        ae_optimizer_args: List[Dict] | None = None,
        trainable_disc_params: List[List[str]] | None = None,
        disc_optimizer_args: List[Dict] | None = None,
        disc_start_iter: int = 0,
        diff_boost_factor: float = 3,
        ckpt_engine: None | str | Dict = None,
        ckpt_path: str | None = None,
        additional_decode_keys: List[str] | None = None,
        trainable: bool | str = True,
        time_n_compress: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            loss_config=loss_config,
            regularizer_config=regularizer_config,
            optimizer_config=optimizer_config,
            lr_g_factor=lr_g_factor,
            lr_g_scheduler_config=lr_g_scheduler_config,
            lr_d_scheduler_config=lr_d_scheduler_config,
            trainable_ae_params=trainable_ae_params,
            ae_optimizer_args=ae_optimizer_args,
            trainable_disc_params=trainable_disc_params,
            disc_optimizer_args=disc_optimizer_args,
            disc_start_iter=disc_start_iter,
            diff_boost_factor=diff_boost_factor,
            ckpt_engine=None,
            ckpt_path=None,
            additional_decode_keys=additional_decode_keys,
            trainable=trainable,
            **kwargs,
        )

        self.constraint_encoder = instantiate_from_config(constraint_encoder_config)
        self.constraint_encoder.requires_grad_(False)

        self.constraint_decoder = instantiate_from_config(constraint_decoder_config)
        self.constraint_decoder.requires_grad_(False)
        if ckpt_path is not None:
            assert ckpt_engine is None, "Can't set ckpt_engine and ckpt_path"
            logpy.warn("Checkpoint path is deprecated, use `checkpoint_egnine` instead")
        self.apply_ckpt(default(ckpt_path, ckpt_engine))
        self.time_n_compress = time_n_compress

    def forward(
        self, x: torch.Tensor, **additional_decode_kwargs
    ) -> Tuple[torch.Tensor | Dict]:
        x_d = x[:, :, :: self.time_n_compress, :, :]
        with torch.no_grad():
            z_d = self.constraint_encoder(x_d)
        z = self.encoder(x)
        z = torch.cat([z, z_d], dim=0)
        z, reg_log = self.regularization(z)
        dec = self.decoder(z, **additional_decode_kwargs)
        x_d = self.constraint_decoder(z[: z.shape[0] // 2], **additional_decode_kwargs)
        return z, dec, reg_log, x_d

    def inner_training_step(
        self, batch: dict, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        x = self.get_input(batch)
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        z, xrec, regularization_log, xrec_2d = self(x, **additional_decode_kwargs)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "train",
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        if optimizer_idx == 0:
            # autoencode
            out_loss = self.loss(x, xrec, xrec_2d, **extra_info)
            if isinstance(out_loss, tuple):
                aeloss, log_dict_ae = out_loss
            else:
                # simple loss function
                aeloss = out_loss
                log_dict_ae = {"train/loss/rec": aeloss.detach()}

            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                "loss",
                aeloss.mean().detach(),
                prog_bar=True,
                logger=False,
                on_epoch=False,
                on_step=True,
            )
            return aeloss
        elif optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, xrec_2d, **extra_info)
            # -> discriminator always needs to return a tuple
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return discloss
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")

    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        x = self.get_input(batch)

        z, xrec, regularization_log, xrec_2d = self(x)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": 0,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "val" + postfix,
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()
        out_loss = self.loss(x, xrec, xrec_2d, **extra_info)
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss
        else:
            # simple loss function
            aeloss = out_loss
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}
        full_log_dict = log_dict_ae

        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1
            discloss, log_dict_disc = self.loss(x, xrec, xrec_2d, **extra_info)
            full_log_dict.update(log_dict_disc)
        self.log(
            f"val{postfix}/loss/rec",
            log_dict_ae[f"val{postfix}/loss/rec"],
            sync_dist=True,
        )
        self.log_dict(full_log_dict, sync_dist=True)
        return full_log_dict

    @torch.no_grad()
    def log_images(
        self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs
    ) -> dict:
        log = dict()
        additional_decode_kwargs = {}
        x = self.get_input(batch)
        additional_decode_kwargs.update(
            {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
        )

        z, xrec, _, xrec_2d = self(x, **additional_decode_kwargs)
        xrec, xrec_d = torch.chunk(xrec, 2)
        is_video = x.dim() == 5
        if is_video:
            b, c, t, h, w = x.shape
            x = rearrange(x, "b c t h w -> (b t) c h w")
            xrec = rearrange(xrec, "b c t h w -> (b t) c h w")
            xrec_d = rearrange(xrec_d, "b c t h w -> (b t) c h w")

        log["inputs"] = x
        log["reconstructions"] = xrec
        log["rec_d"] = xrec_d

        diff = 0.5 * torch.abs(torch.clamp(xrec, -1.0, 1.0) - x)
        diff.clamp_(0, 1.0)
        log["diff"] = 2.0 * diff - 1.0
        # diff_boost shows location of small errors, by boosting their
        # brightness.
        log["diff_boost"] = (
            2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1
        )
        if is_video:
            x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
            xrec = rearrange(xrec, "(b t) c h w -> b c t h w", b=b, t=t)
        # if hasattr(self.loss, "log_images"):
        #     log.update(self.loss.log_images(x, xrec))
        with self.ema_scope():
            _, xrec_ema, _, _ = self(x, **additional_decode_kwargs)
            xrec_ema, _ = torch.chunk(xrec_ema, 2)

        if is_video:
            x = rearrange(x, "b c t h w -> (b t) c h w")
            xrec_ema = rearrange(xrec_ema, "b c t h w -> (b t) c h w")

            log["reconstructions_ema"] = xrec_ema
            diff_ema = 0.5 * torch.abs(torch.clamp(xrec_ema, -1.0, 1.0) - x)
            diff_ema.clamp_(0, 1.0)
            log["diff_ema"] = 2.0 * diff_ema - 1.0
            log["diff_boost_ema"] = (
                2.0 * torch.clamp(self.diff_boost_factor * diff_ema, 0.0, 1.0) - 1
            )
        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            if is_video:
                x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
            _, xrec_add, _, _ = self(x, **additional_decode_kwargs)
            xrec_add, _ = torch.chunk(xrec_add, 2)
            if is_video:
                x = rearrange(x, "b c t h w -> (b t) c h w")
                xrec_add = rearrange(xrec_add, "b c t h w -> (b t) c h w")
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            log[log_str] = xrec_add
        return log
