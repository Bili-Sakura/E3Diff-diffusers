"""CUT inference pipeline following ``pytorch-image-translation-models`` style.

This pipeline wraps a pre-trained CUT generator for end-to-end
inference.  It mirrors the :class:`I2SBPipeline` pattern from the
library: accept a source tensor, run a single forward pass through the
generator, and return results in the requested format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models.generators import ResNetGenerator


@dataclass
class CUTPipelineOutput:
    """Container for CUT pipeline results.

    Attributes
    ----------
    images:
        Translated images in the requested format (``torch.Tensor``,
        ``list[PIL.Image.Image]``, or ``np.ndarray``).
    """

    images: Any


class CUTPipeline:
    """End-to-end inference pipeline for a CUT generator.

    Parameters
    ----------
    generator:
        A :class:`ResNetGenerator` (or compatible ``nn.Module``) with
        pre-trained weights.

    Example
    -------
    >>> from src.models.generators import ResNetGenerator
    >>> gen = ResNetGenerator(in_channels=3, out_channels=3)
    >>> pipeline = CUTPipeline(gen)
    >>> source = torch.randn(1, 3, 256, 256)
    >>> result = pipeline(source, output_type="pt")
    >>> result.images.shape
    torch.Size([1, 3, 256, 256])
    """

    def __init__(self, generator: ResNetGenerator) -> None:
        self.generator = generator

    @torch.no_grad()
    def __call__(
        self,
        source: torch.Tensor,
        output_type: str = "pt",
    ) -> CUTPipelineOutput:
        """Run the CUT generator on *source* images.

        Parameters
        ----------
        source:
            Source images ``[B, C, H, W]``.
        output_type:
            ``"pt"`` for tensors, ``"pil"`` for PIL images, ``"np"``
            for NumPy arrays.
        """
        self.generator.eval()
        output = self.generator(source)
        return self._format_output(output, output_type)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 64,
        n_residual_blocks: int = 9,
        device: str = "cpu",
    ) -> "CUTPipeline":
        """Build a pipeline from a saved CUT generator checkpoint.

        Parameters
        ----------
        checkpoint_path:
            Path to a ``.pth`` file containing the generator
            ``state_dict``.
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        base_filters:
            Base filter count for the ResNet generator.
        n_residual_blocks:
            Number of residual blocks.
        device:
            Target device string.
        """
        generator = ResNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            n_residual_blocks=n_residual_blocks,
        )

        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "generator" in state:
            state = state["generator"]
        generator.load_state_dict(state, strict=False)
        generator = generator.to(device).eval()
        return cls(generator)

    # ------------------------------------------------------------------
    # Output formatting (mirrors I2SBPipeline._format_output)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(
        tensor: torch.Tensor,
        output_type: str,
    ) -> CUTPipelineOutput:
        if output_type == "pt":
            return CUTPipelineOutput(images=tensor)

        if output_type == "np":
            return CUTPipelineOutput(images=tensor.cpu().numpy())

        if output_type == "pil":
            images: list[Image.Image] = []
            arr = tensor.clamp(-1, 1).cpu()
            arr = (arr + 1.0) / 2.0  # [-1, 1] → [0, 1]
            for i in range(arr.shape[0]):
                img_np = arr[i].permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                if img_np.shape[2] == 1:
                    img_np = img_np[:, :, 0]
                images.append(Image.fromarray(img_np))
            return CUTPipelineOutput(images=images)

        raise ValueError(f"Unknown output_type: {output_type!r}")
