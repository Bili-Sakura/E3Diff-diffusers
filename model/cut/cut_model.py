"""CUT model using ``pytorch-image-translation-models`` library components.

This module wraps the library's :class:`ResNetGenerator` and
:class:`PatchGANDiscriminator` to implement *Contrastive Learning for
Unpaired Image-to-Image Translation* (CUT, ECCV 2020).

Checkpoint loading follows the :class:`ImageTranslator` pattern from the
library so that pre-trained CUT weights can be loaded and used for
inference directly.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from src.inference.predictor import ImageTranslator
from src.models.generators import ResNetGenerator
from src.models.discriminators import PatchGANDiscriminator

from model.base_model import BaseModel
from model.cut.patchnce_loss import PatchNCELoss, PatchSampleMLP

logger = logging.getLogger("base")


def define_cut_generator(opt: dict) -> ResNetGenerator:
    """Instantiate a CUT generator from a config dict.

    Parameters
    ----------
    opt:
        Model configuration.  Expected keys under ``opt["model"]["cut"]``:

        * ``in_channels`` (int, default 3)
        * ``out_channels`` (int, default 3)
        * ``base_filters`` (int, default 64)
        * ``n_residual_blocks`` (int, default 9)
    """
    cut_opt = opt.get("model", {}).get("cut", {})
    return ResNetGenerator(
        in_channels=cut_opt.get("in_channels", 3),
        out_channels=cut_opt.get("out_channels", 3),
        base_filters=cut_opt.get("base_filters", 64),
        n_residual_blocks=cut_opt.get("n_residual_blocks", 9),
    )


def define_cut_discriminator(opt: dict) -> PatchGANDiscriminator:
    """Instantiate a CUT discriminator from a config dict.

    Parameters
    ----------
    opt:
        Model configuration.  Expected keys under ``opt["model"]["cut"]``:

        * ``disc_in_channels`` (int, default 3)
        * ``disc_base_filters`` (int, default 64)
        * ``disc_n_layers`` (int, default 3)
    """
    cut_opt = opt.get("model", {}).get("cut", {})
    return PatchGANDiscriminator(
        in_channels=cut_opt.get("disc_in_channels", 3),
        base_filters=cut_opt.get("disc_base_filters", 64),
        n_layers=cut_opt.get("disc_n_layers", 3),
    )


class CUTModel(BaseModel):
    """CUT model for unpaired image-to-image translation.

    Uses the ``pytorch-image-translation-models`` library for the
    generator (:class:`ResNetGenerator`) and discriminator
    (:class:`PatchGANDiscriminator`) architectures.

    The generator can be loaded from a pre-trained CUT checkpoint
    via :meth:`load_cut_checkpoint` or the library's
    :class:`ImageTranslator`.

    Parameters
    ----------
    opt:
        Full configuration dictionary.  CUT-specific keys live under
        ``opt["model"]["cut"]``.
    """

    def __init__(self, opt: dict) -> None:
        super().__init__(opt)

        # -- Generator (ResNet from the library) --------------------------
        self.netG: ResNetGenerator = self.set_device(define_cut_generator(opt))

        cut_opt = opt.get("model", {}).get("cut", {})

        # -- Discriminator (PatchGAN from the library) --------------------
        self.netD: PatchGANDiscriminator = self.set_device(
            define_cut_discriminator(opt)
        )

        # -- MLP heads for PatchNCE (encoder layers → embeddings) ---------
        nce_layers = cut_opt.get("nce_layers", [0, 4, 8, 12, 16])
        if not nce_layers:
            raise ValueError("nce_layers must be a non-empty list of layer indices")
        mlp_channels = cut_opt.get("mlp_channels", 256)
        num_patches = cut_opt.get("num_patches", 256)
        self.nce_layers = nce_layers
        self.mlp_heads: nn.ModuleList = nn.ModuleList()
        # Placeholder — actual channel sizes depend on the generator's
        # encoder; they are lazily initialised on the first forward pass
        # when ``_init_mlp_heads`` is called.
        self._mlp_init_done = False
        self._mlp_channels = mlp_channels
        self._num_patches = num_patches

        # -- Losses -------------------------------------------------------
        self.nce_loss = PatchNCELoss(
            temperature=cut_opt.get("nce_temperature", 0.07),
        )
        self.gan_loss = nn.MSELoss()  # LSGAN by default
        self.lambda_nce = cut_opt.get("lambda_nce", 1.0)
        self.lambda_gan = cut_opt.get("lambda_gan", 1.0)
        self.nce_idt = cut_opt.get("nce_idt", True)

        # -- Optimizers ---------------------------------------------------
        self.log_dict: OrderedDict[str, float] = OrderedDict()
        if opt.get("phase") == "train":
            lr = opt.get("train", {}).get("optimizer", {}).get("lr", 2e-4)
            beta1 = cut_opt.get("beta1", 0.5)

            self.optG = torch.optim.Adam(
                self.netG.parameters(), lr=lr, betas=(beta1, 0.999)
            )
            self.optD = torch.optim.Adam(
                self.netD.parameters(), lr=lr, betas=(beta1, 0.999)
            )
            self.netG.train()
            self.netD.train()

        # -- Load checkpoint if provided ----------------------------------
        self.load_network()

    # ------------------------------------------------------------------
    # Encoder feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_encoder_features(
        generator: ResNetGenerator,
        x: torch.Tensor,
        layers: list[int],
    ) -> list[torch.Tensor]:
        """Run *x* through the generator's ``model`` sequential and
        return intermediate feature maps at the given layer indices.
        """
        feats: list[torch.Tensor] = []
        h = x
        for i, layer in enumerate(generator.model):
            h = layer(h)
            if i in layers:
                feats.append(h)
        return feats

    def _init_mlp_heads(self, feats: list[torch.Tensor]) -> None:
        """Lazily build one MLP head per extracted feature layer."""
        for feat in feats:
            in_ch = feat.shape[1]
            head = PatchSampleMLP(
                in_channels=in_ch,
                out_channels=self._mlp_channels,
                num_patches=self._num_patches,
            )
            self.mlp_heads.append(self.set_device(head))
        self._mlp_init_done = True

        # Add MLP params to the generator optimizer
        if hasattr(self, "optG"):
            for head in self.mlp_heads:
                self.optG.add_param_group(
                    {"params": head.parameters(), "lr": self.optG.defaults["lr"]}
                )

    # ------------------------------------------------------------------
    # Data / forward / optimisation
    # ------------------------------------------------------------------

    def feed_data(self, data: dict) -> None:
        """Accept a data dict with keys ``"source"`` and ``"target"``."""
        self.real_A = self.set_device(data["source"])
        self.real_B = self.set_device(data["target"])

    def forward_G(self) -> torch.Tensor:
        """Generator forward: source → fake target."""
        self.fake_B = self.netG(self.real_A)
        return self.fake_B

    def optimize_parameters(self) -> None:
        """One training step (generator + discriminator)."""
        # ---- Generator & NCE ----
        self.optG.zero_grad()
        fake_B = self.forward_G()

        # GAN loss
        pred_fake = self.netD(fake_B)
        target_real = torch.ones_like(pred_fake)
        loss_G_gan = self.gan_loss(pred_fake, target_real) * self.lambda_gan

        # PatchNCE loss
        loss_nce = self._compute_nce_loss(self.real_A, fake_B)

        loss_nce_idt = torch.tensor(0.0, device=self.device)
        if self.nce_idt:
            idt_B = self.netG(self.real_B)
            loss_nce_idt = self._compute_nce_loss(self.real_B, idt_B)

        loss_G = loss_G_gan + loss_nce + loss_nce_idt
        loss_G.backward()
        self.optG.step()

        # ---- Discriminator ----
        self.optD.zero_grad()
        pred_real = self.netD(self.real_B)
        target_real = torch.ones_like(pred_real)
        loss_D_real = self.gan_loss(pred_real, target_real)

        pred_fake_d = self.netD(fake_B.detach())
        target_fake = torch.zeros_like(pred_fake_d)
        loss_D_fake = self.gan_loss(pred_fake_d, target_fake)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.optD.step()

        # ---- Logging ----
        self.log_dict["l_G_gan"] = loss_G_gan.item()
        self.log_dict["l_nce"] = loss_nce.item()
        self.log_dict["l_nce_idt"] = loss_nce_idt.item()
        self.log_dict["l_D"] = loss_D.item()

    def _compute_nce_loss(
        self,
        source: torch.Tensor,
        generated: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PatchNCE loss between source and generated images."""
        feats_src = self._extract_encoder_features(
            self.netG, source, self.nce_layers
        )
        feats_gen = self._extract_encoder_features(
            self.netG, generated, self.nce_layers
        )

        if not self._mlp_init_done:
            self._init_mlp_heads(feats_src)

        total_loss = torch.tensor(0.0, device=self.device)
        for feat_s, feat_g, mlp in zip(feats_src, feats_gen, self.mlp_heads):
            emb_s, patch_ids = mlp(feat_s)
            emb_g, _ = mlp(feat_g, patch_ids=patch_ids)
            total_loss = total_loss + self.nce_loss(emb_g, emb_s.detach())

        return total_loss / len(self.nce_layers)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test(self, **kwargs) -> None:
        """Generate translated images (inference mode)."""
        self.netG.eval()
        self.fake_B = self.netG(self.real_A)
        self.netG.train()

    def get_current_log(self) -> OrderedDict:
        return self.log_dict

    def get_current_visuals(self, **kwargs) -> OrderedDict:
        out = OrderedDict()
        out["source"] = self.real_A.detach().cpu()
        out["target"] = self.real_B.detach().cpu()
        out["generated"] = self.fake_B.detach().cpu()
        return out

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_network(self, epoch: int, iter_step: int) -> None:
        """Save generator and discriminator checkpoints."""
        ckpt_dir = self.opt["path"]["checkpoint"]
        gen_path = os.path.join(ckpt_dir, f"I{iter_step}_E{epoch}_cut_gen.pth")
        disc_path = os.path.join(ckpt_dir, f"I{iter_step}_E{epoch}_cut_disc.pth")

        torch.save(self.netG.state_dict(), gen_path)
        torch.save(
            {
                "discriminator": self.netD.state_dict(),
                "optimizer_g": self.optG.state_dict(),
                "optimizer_d": self.optD.state_dict(),
                "epoch": epoch,
                "iter": iter_step,
            },
            disc_path,
        )
        logger.info("Saved CUT model in [%s]", gen_path)

    def load_network(self) -> None:
        """Load a CUT checkpoint if ``resume_state`` is set."""
        load_path = self.opt.get("path", {}).get("resume_state")
        if load_path is None:
            return

        gen_path = f"{load_path}_cut_gen.pth"
        if os.path.isfile(gen_path):
            logger.info("Loading CUT generator from [%s]", gen_path)
            state = torch.load(gen_path, map_location=self.device, weights_only=True)
            self.netG.load_state_dict(state, strict=False)

        disc_path = f"{load_path}_cut_disc.pth"
        if os.path.isfile(disc_path) and self.opt.get("phase") == "train":
            logger.info("Loading CUT discriminator from [%s]", disc_path)
            # weights_only=False is required for optimizer state dicts;
            # only load checkpoints from trusted sources.
            ckpt = torch.load(disc_path, map_location=self.device, weights_only=False)
            self.netD.load_state_dict(ckpt["discriminator"], strict=False)
            self.optG.load_state_dict(ckpt["optimizer_g"])
            self.optD.load_state_dict(ckpt["optimizer_d"])
            self.begin_step = ckpt.get("iter", 0)
            self.begin_epoch = ckpt.get("epoch", 0)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        opt: dict | None = None,
        device: str = "cpu",
    ) -> "CUTModel":
        """Load a pre-trained CUT model for inference.

        This is a convenience wrapper around the library's
        :class:`ImageTranslator` pattern.

        Parameters
        ----------
        checkpoint_path:
            Path to the generator ``.pth`` file.
        opt:
            Optional config dict.  If ``None`` a minimal default is used.
        device:
            Device string (``"cpu"`` or ``"cuda"``).
        """
        if opt is None:
            opt = {
                "gpu_ids": None if device == "cpu" else [0],
                "phase": "val",
                "model": {"cut": {}},
                "path": {"resume_state": None},
            }

        model = cls.__new__(cls)
        BaseModel.__init__(model, opt)
        model.netG = define_cut_generator(opt).to(model.device)

        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Support both raw state_dict and nested {"generator": ...} format
        if "generator" in state:
            state = state["generator"]
        model.netG.load_state_dict(state, strict=False)
        model.netG.eval()
        model.log_dict = OrderedDict()
        logger.info("Loaded CUT generator from %s", checkpoint_path)
        return model

    def get_translator(self, image_size: int = 256) -> ImageTranslator:
        """Return a library :class:`ImageTranslator` for easy inference.

        Parameters
        ----------
        image_size:
            Target spatial resolution for the input transform.
        """
        return ImageTranslator(
            generator=self.netG,
            device=str(self.device),
            image_size=image_size,
        )

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_network(self) -> None:
        s, n = self.get_network_description(self.netG)
        net_struc_str = self.netG.__class__.__name__
        logger.info(
            "CUT Generator structure: %s, with parameters: %d", net_struc_str, n
        )
