"""Tests for CUT pipeline and PatchNCE loss."""

import numpy as np
import torch
from PIL import Image

from src.models.generators import ResNetGenerator

from model.cut.cut_pipeline import CUTPipeline, CUTPipelineOutput
from model.cut.patchnce_loss import PatchNCELoss, PatchSampleMLP


# -- PatchSampleMLP -------------------------------------------------------

class TestPatchSampleMLP:
    def test_output_shape(self):
        mlp = PatchSampleMLP(in_channels=64, out_channels=32, num_patches=16)
        feat = torch.randn(2, 64, 8, 8)
        sample, patch_ids = mlp(feat)
        assert sample.shape == (2, 16, 32)
        assert patch_ids.shape == (2, 16)

    def test_reuse_patch_ids(self):
        mlp = PatchSampleMLP(in_channels=64, out_channels=32, num_patches=16)
        feat = torch.randn(2, 64, 8, 8)
        _, ids = mlp(feat)
        sample2, ids2 = mlp(feat, patch_ids=ids)
        assert torch.equal(ids, ids2)
        assert sample2.shape == (2, 16, 32)

    def test_l2_normalised(self):
        mlp = PatchSampleMLP(in_channels=64, out_channels=32, num_patches=16)
        feat = torch.randn(2, 64, 8, 8)
        sample, _ = mlp(feat)
        norms = torch.norm(sample, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_fewer_patches_than_spatial(self):
        mlp = PatchSampleMLP(in_channels=64, out_channels=32, num_patches=4)
        feat = torch.randn(1, 64, 2, 2)  # only 4 spatial positions
        sample, ids = mlp(feat)
        assert sample.shape[1] == 4


# -- PatchNCELoss ---------------------------------------------------------

class TestPatchNCELoss:
    def test_basic_loss(self):
        loss_fn = PatchNCELoss(temperature=0.07)
        feat_q = torch.randn(2, 16, 32)
        feat_k = torch.randn(2, 16, 32)
        loss = loss_fn(feat_q, feat_k)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_identical_features_low_loss(self):
        loss_fn = PatchNCELoss(temperature=0.07)
        feat = torch.randn(2, 16, 32)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        loss = loss_fn(feat, feat)
        # When query == key, loss should be relatively low
        assert loss.item() < 5.0

    def test_all_negatives_from_minibatch(self):
        loss_fn = PatchNCELoss(
            temperature=0.07,
            nce_includes_all_negatives_from_minibatch=True,
        )
        feat_q = torch.randn(2, 8, 32)
        feat_k = torch.randn(2, 8, 32)
        loss = loss_fn(feat_q, feat_k)
        assert loss.dim() == 0


# -- CUTPipeline -----------------------------------------------------------

class TestCUTPipeline:
    def _make_pipeline(self):
        gen = ResNetGenerator(
            in_channels=3, out_channels=3,
            base_filters=16, n_residual_blocks=2,
        )
        return CUTPipeline(gen)

    def test_pt_output(self):
        pipeline = self._make_pipeline()
        source = torch.randn(1, 3, 64, 64)
        result = pipeline(source, output_type="pt")
        assert isinstance(result, CUTPipelineOutput)
        assert isinstance(result.images, torch.Tensor)
        assert result.images.shape == (1, 3, 64, 64)

    def test_np_output(self):
        pipeline = self._make_pipeline()
        source = torch.randn(1, 3, 64, 64)
        result = pipeline(source, output_type="np")
        assert isinstance(result.images, np.ndarray)
        assert result.images.shape == (1, 3, 64, 64)

    def test_pil_output(self):
        pipeline = self._make_pipeline()
        source = torch.randn(1, 3, 64, 64)
        result = pipeline(source, output_type="pil")
        assert isinstance(result.images, list)
        assert len(result.images) == 1
        assert isinstance(result.images[0], Image.Image)

    def test_batch_pil_output(self):
        pipeline = self._make_pipeline()
        source = torch.randn(3, 3, 64, 64)
        result = pipeline(source, output_type="pil")
        assert len(result.images) == 3

    def test_invalid_output_type_raises(self):
        import pytest

        pipeline = self._make_pipeline()
        source = torch.randn(1, 3, 64, 64)
        with pytest.raises(ValueError, match="invalid"):
            pipeline(source, output_type="invalid")

    def test_from_checkpoint(self, tmp_path):
        gen = ResNetGenerator(
            in_channels=3, out_channels=3,
            base_filters=16, n_residual_blocks=2,
        )
        ckpt_path = tmp_path / "gen.pth"
        torch.save(gen.state_dict(), ckpt_path)

        pipeline = CUTPipeline.from_checkpoint(
            str(ckpt_path),
            base_filters=16,
            n_residual_blocks=2,
        )
        source = torch.randn(1, 3, 64, 64)
        result = pipeline(source, output_type="pt")
        assert result.images.shape == (1, 3, 64, 64)

    def test_from_checkpoint_nested_format(self, tmp_path):
        gen = ResNetGenerator(
            in_channels=3, out_channels=3,
            base_filters=16, n_residual_blocks=2,
        )
        ckpt_path = tmp_path / "gen_nested.pth"
        torch.save({"generator": gen.state_dict()}, ckpt_path)

        pipeline = CUTPipeline.from_checkpoint(
            str(ckpt_path),
            base_filters=16,
            n_residual_blocks=2,
        )
        source = torch.randn(1, 3, 64, 64)
        result = pipeline(source, output_type="pt")
        assert result.images.shape == (1, 3, 64, 64)
