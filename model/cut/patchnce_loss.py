"""PatchNCE contrastive loss for CUT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSampleMLP(nn.Module):
    """MLP head that projects encoder feature patches into an embedding space.

    Each spatial location in the feature map is treated as a separate
    patch.  The MLP maps each patch vector to a lower-dimensional,
    L2-normalised embedding used for the InfoNCE contrastive loss.

    Parameters
    ----------
    in_channels:
        Number of input feature channels.
    out_channels:
        Embedding dimensionality.
    num_patches:
        Maximum number of patches to sample per feature map.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        num_patches: int = 256,
    ) -> None:
        super().__init__()
        self.l2norm = Normalize(2)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
        self.num_patches = num_patches

    def forward(
        self,
        feat: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and sample patches from a feature map.

        Parameters
        ----------
        feat:
            Feature map ``[B, C, H, W]``.
        patch_ids:
            Pre-selected spatial indices ``[B, num_patches]``.
            If ``None``, indices are randomly sampled.

        Returns
        -------
        sample:
            L2-normalised patch embeddings ``[B, num_patches, out_channels]``.
        patch_ids:
            The spatial indices used for sampling.
        """
        B, C, H, W = feat.shape
        feat_reshape = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        if patch_ids is None:
            num_patches = min(self.num_patches, H * W)
            patch_ids = torch.stack(
                [torch.randperm(H * W, device=feat.device)[:num_patches] for _ in range(B)],
                dim=0,
            )

        feat_sampled = torch.gather(
            feat_reshape,
            1,
            patch_ids.unsqueeze(-1).expand(-1, -1, C),
        )
        sample = self.mlp(feat_sampled)
        sample = self.l2norm(sample)
        return sample, patch_ids


class Normalize(nn.Module):
    """L2 normalisation along a given dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=self.dim)


class PatchNCELoss(nn.Module):
    """Patch-wise noise-contrastive estimation loss for CUT.

    For every query patch in the generated image, the positive is the
    corresponding patch in the *source* image at the same spatial
    location.  All other patches in the batch serve as negatives.

    Parameters
    ----------
    temperature:
        Temperature scaling for the softmax.
    nce_includes_all_negatives_from_minibatch:
        If ``True``, negatives are drawn from the entire mini-batch
        rather than only from the same image.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        nce_includes_all_negatives_from_minibatch: bool = False,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.nce_includes_all = nce_includes_all_negatives_from_minibatch
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PatchNCE loss.

        Parameters
        ----------
        feat_q:
            Query embeddings ``[B, N, C]`` (from generated image).
        feat_k:
            Key embeddings ``[B, N, C]`` (from source image).

        Returns
        -------
        torch.Tensor:
            Scalar loss value.
        """
        B, N, C = feat_q.shape

        # Positive logit: dot product between query and its paired key
        l_pos = (feat_q * feat_k).sum(dim=-1, keepdim=True)  # [B, N, 1]

        if self.nce_includes_all:
            # All keys from the whole batch as negatives
            feat_k_neg = feat_k.reshape(-1, C)  # [B*N, C]
            l_neg = torch.mm(
                feat_q.reshape(-1, C), feat_k_neg.T
            )  # [B*N, B*N]
            l_neg = l_neg.reshape(B, N, -1)  # [B, N, B*N]
        else:
            # Negatives from the same image only
            l_neg = torch.bmm(feat_q, feat_k.permute(0, 2, 1))  # [B, N, N]

        logits = torch.cat([l_pos, l_neg], dim=-1) / self.temperature  # [B, N, 1+K]

        # The positive is always at index 0
        labels = torch.zeros(B * N, dtype=torch.long, device=feat_q.device)
        loss = self.cross_entropy(logits.reshape(-1, logits.size(-1)), labels)
        return loss.mean()
