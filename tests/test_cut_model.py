"""Tests for CUT model components and integration with pytorch-image-translation-models."""

import torch
from collections import OrderedDict

from src.models.generators import ResNetGenerator
from src.models.discriminators import PatchGANDiscriminator

from model.cut.cut_model import (
    CUTModel,
    define_cut_generator,
    define_cut_discriminator,
)


# -- Generator factory ---------------------------------------------------

class TestDefineCutGenerator:
    def test_default_config(self):
        opt = {"model": {"cut": {}}}
        gen = define_cut_generator(opt)
        assert isinstance(gen, ResNetGenerator)

    def test_custom_channels(self):
        opt = {"model": {"cut": {"in_channels": 1, "out_channels": 4}}}
        gen = define_cut_generator(opt)
        x = torch.randn(1, 1, 64, 64)
        y = gen(x)
        assert y.shape == (1, 4, 64, 64)

    def test_forward_shape(self):
        opt = {"model": {"cut": {"base_filters": 16, "n_residual_blocks": 2}}}
        gen = define_cut_generator(opt)
        x = torch.randn(1, 3, 64, 64)
        y = gen(x)
        assert y.shape == (1, 3, 64, 64)

    def test_output_range(self):
        opt = {"model": {"cut": {"base_filters": 16, "n_residual_blocks": 2}}}
        gen = define_cut_generator(opt)
        x = torch.randn(1, 3, 64, 64)
        y = gen(x)
        assert y.min() >= -1.0
        assert y.max() <= 1.0


# -- Discriminator factory ------------------------------------------------

class TestDefineCutDiscriminator:
    def test_default_config(self):
        opt = {"model": {"cut": {}}}
        disc = define_cut_discriminator(opt)
        assert isinstance(disc, PatchGANDiscriminator)

    def test_spatial_output(self):
        opt = {"model": {"cut": {"disc_in_channels": 3, "disc_base_filters": 16, "disc_n_layers": 2}}}
        disc = define_cut_discriminator(opt)
        x = torch.randn(1, 3, 64, 64)
        y = disc(x)
        assert y.dim() == 4
        assert y.shape[0] == 1
        assert y.shape[1] == 1


# -- CUTModel ------------------------------------------------------------

def _make_cut_opt(phase="val", resume_state=None):
    """Build a minimal CUT config dict for testing."""
    return {
        "gpu_ids": None,
        "phase": phase,
        "model": {
            "which_model_G": "cut",
            "cut": {
                "in_channels": 3,
                "out_channels": 3,
                "base_filters": 16,
                "n_residual_blocks": 2,
                "disc_in_channels": 3,
                "disc_base_filters": 16,
                "disc_n_layers": 2,
                "nce_layers": [0, 4],
                "mlp_channels": 32,
                "num_patches": 16,
                "nce_temperature": 0.07,
                "lambda_nce": 1.0,
                "lambda_gan": 1.0,
                "nce_idt": False,
                "beta1": 0.5,
            },
        },
        "path": {"resume_state": resume_state, "checkpoint": "/tmp/ckpt"},
        "train": {"optimizer": {"lr": 1e-4}},
    }


class TestCUTModel:
    def test_creates_generator_and_discriminator(self):
        opt = _make_cut_opt(phase="val")
        model = CUTModel(opt)
        assert isinstance(model.netG, ResNetGenerator)
        assert isinstance(model.netD, PatchGANDiscriminator)

    def test_test_forward(self):
        opt = _make_cut_opt(phase="val")
        model = CUTModel(opt)
        model.real_A = torch.randn(1, 3, 64, 64)
        model.real_B = torch.randn(1, 3, 64, 64)
        model.test()
        assert model.fake_B.shape == (1, 3, 64, 64)

    def test_get_current_visuals(self):
        opt = _make_cut_opt(phase="val")
        model = CUTModel(opt)
        model.real_A = torch.randn(1, 3, 64, 64)
        model.real_B = torch.randn(1, 3, 64, 64)
        model.test()
        visuals = model.get_current_visuals()
        assert "source" in visuals
        assert "target" in visuals
        assert "generated" in visuals

    def test_optimize_parameters(self):
        opt = _make_cut_opt(phase="train")
        model = CUTModel(opt)
        data = {
            "source": torch.randn(2, 3, 64, 64),
            "target": torch.randn(2, 3, 64, 64),
        }
        model.feed_data(data)
        model.optimize_parameters()
        log = model.get_current_log()
        assert "l_G_gan" in log
        assert "l_nce" in log
        assert "l_D" in log

    def test_get_translator(self):
        from src.inference.predictor import ImageTranslator

        opt = _make_cut_opt(phase="val")
        model = CUTModel(opt)
        translator = model.get_translator(image_size=64)
        assert isinstance(translator, ImageTranslator)

    def test_save_and_load(self, tmp_path):
        opt = _make_cut_opt(phase="train")
        opt["path"]["checkpoint"] = str(tmp_path)
        model = CUTModel(opt)

        model.save_network(epoch=1, iter_step=100)

        gen_path = tmp_path / "I100_E1_cut_gen.pth"
        disc_path = tmp_path / "I100_E1_cut_disc.pth"
        assert gen_path.exists()
        assert disc_path.exists()

        # Load into a new model
        opt2 = _make_cut_opt(phase="train", resume_state=str(tmp_path / "I100_E1"))
        model2 = CUTModel(opt2)
        assert model2.begin_step == 100
        assert model2.begin_epoch == 1


# -- Model factory --------------------------------------------------------

class TestModelFactory:
    def test_create_cut_model(self):
        from model import create_model

        opt = _make_cut_opt(phase="val")
        m = create_model(opt)
        assert isinstance(m, CUTModel)
