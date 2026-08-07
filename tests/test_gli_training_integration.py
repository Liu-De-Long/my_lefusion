from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "LeFusion"))

from ddpm import (  # noqa: E402
    GaussianDiffusion_Nolatent,
    masked_lesion_loss,
    soft_histogram_loss_details,
    normalize_spatial_shape,
    prepare_gli_spatial_condition,
    prepare_training_batch,
    Unet3D,
)
from train.train import validate_training_config  # noqa: E402


class _ZeroDenoiser(torch.nn.Module):
    def forward(self, x, time, cond=None, **kwargs):
        return torch.zeros_like(x)


class GLITrainingIntegrationTests(unittest.TestCase):
    def test_non_empty_sample_channel_units_are_equally_weighted(self) -> None:
        prediction = torch.zeros((2, 4, 1, 1, 2))
        target = torch.zeros_like(prediction)
        mask = torch.zeros_like(prediction)
        target[0, 0, 0, 0, 0] = 2.0
        mask[0, 0, 0, 0, 0] = 1.0
        target[1, 3, 0, 0, :] = 1.0
        mask[1, 3, 0, 0, :] = 1.0

        self.assertAlmostEqual(masked_lesion_loss(prediction, target, mask, 'l1').item(), 1.5)
        self.assertAlmostEqual(masked_lesion_loss(prediction, target, mask, 'l2').item(), 2.5)

    def test_empty_units_are_skipped_and_all_empty_batch_fails(self) -> None:
        prediction = torch.zeros((1, 4, 1, 1, 2))
        target = torch.ones_like(prediction)
        mask = torch.zeros_like(prediction)
        mask[:, 2, :, :, 0] = 1.0
        self.assertAlmostEqual(masked_lesion_loss(prediction, target, mask).item(), 1.0)
        with self.assertRaisesRegex(ValueError, "no lesion voxels"):
            masked_lesion_loss(prediction, target, torch.zeros_like(mask))

    def test_single_t1c_prediction_is_masked_by_four_labels_without_state_duplication(self) -> None:
        prediction = torch.zeros((1, 1, 1, 1, 2), requires_grad=True)
        target = torch.tensor([[[[[2.0, 1.0]]]]])
        mask = torch.zeros((1, 4, 1, 1, 2))
        mask[:, 0, :, :, 0] = 1
        mask[:, 3, :, :, 1] = 1
        loss = masked_lesion_loss(prediction, target, mask)
        self.assertAlmostEqual(loss.item(), 1.5)
        loss.backward()
        self.assertNotEqual(float(prediction.grad[0, 0, 0, 0, 0]), 0.0)

        outside_prediction = torch.zeros((1, 1, 1, 1, 3), requires_grad=True)
        outside_target = torch.ones_like(outside_prediction)
        outside_mask = torch.zeros((1, 4, 1, 1, 3))
        outside_mask[:, 1, :, :, 1] = 1
        masked_lesion_loss(outside_prediction, outside_target, outside_mask).backward()
        self.assertEqual(float(outside_prediction.grad[0, 0, 0, 0, 0]), 0.0)
        self.assertNotEqual(float(outside_prediction.grad[0, 0, 0, 0, 1]), 0.0)
        self.assertEqual(float(outside_prediction.grad[0, 0, 0, 0, 2]), 0.0)

    def test_shape_and_fixed_noise_forward_contract(self) -> None:
        diffusion = GaussianDiffusion_Nolatent(
            _ZeroDenoiser(),
            image_size=5,
            num_frames=3,
            spatial_shape=(3, 4, 5),
            channels=4,
            timesteps=4,
            loss_type='l1',
            data_type='gli',
        )
        data = torch.zeros((1, 4, 3, 4, 5))
        mask = torch.zeros_like(data)
        mask[:, 0, 0, 0, 0] = 1
        loss = diffusion(
            x=data,
            mask=mask,
            t=torch.tensor([1]),
            noise=torch.ones_like(data),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(diffusion.sample_shape(2), (2, 4, 3, 4, 5))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            diffusion(x=torch.zeros((1, 4, 3, 5, 5)), mask=mask)

    def test_training_batch_uses_lesion_mask_and_current_device(self) -> None:
        data = torch.zeros((1, 4, 2, 3, 4))
        lesion_mask = torch.ones_like(data)
        label = torch.zeros((1, 1, 2, 3, 4))
        hist = torch.ones((1, 64), dtype=torch.float64)
        moved_data, moved_mask, moved_hist = prepare_training_batch(
            {"data": data, "label": label, "lesion_mask": lesion_mask, "hist": hist},
            torch.device("cpu"),
            "gli",
        )
        self.assertIs(moved_data, data)
        self.assertIs(moved_mask, lesion_mask)
        self.assertEqual(moved_hist.dtype, torch.float32)
        self.assertEqual(moved_hist.device, moved_data.device)

        target_t1c = torch.full((1, 1, 2, 3, 4), 7.0)
        single_data, single_mask, _ = prepare_training_batch(
            {
                "data": data,
                "target_t1c": target_t1c,
                "label": label,
                "lesion_mask": lesion_mask,
                "hist": hist,
            },
            torch.device("cpu"),
            "gli",
            diffusion_channels=1,
        )
        self.assertIs(single_data, target_t1c)
        self.assertIs(single_mask, lesion_mask)

    def test_spatial_condition_contains_hole_and_four_masks(self) -> None:
        lesion_mask = torch.zeros((1, 4, 2, 3, 4))
        lesion_mask[:, 2, 0, 1, 2] = 1
        context = torch.ones((1, 1, 2, 3, 4))
        context[:, :, 0, 1, 2] = 0
        spatial_condition = prepare_gli_spatial_condition(
            {"masked_context": context, "lesion_mask": lesion_mask},
            torch.device("cpu"),
        )
        self.assertEqual(tuple(spatial_condition.shape), (1, 5, 2, 3, 4))
        self.assertTrue(torch.equal(spatial_condition[:, :1], context))
        self.assertTrue(torch.equal(spatial_condition[:, 1:], lesion_mask))

    def test_conditional_unet_has_nine_input_and_four_output_channels(self) -> None:
        model = Unet3D(
            dim=8,
            dim_mults=(1,),
            channels=4,
            spatial_condition_channels=5,
            cond_dim=64,
            attn_heads=1,
            attn_dim_head=8,
            resnet_groups=1,
        )
        self.assertEqual(model.init_conv.in_channels, 9)
        state = torch.zeros((1, 4, 2, 4, 4))
        spatial_condition = torch.zeros((1, 5, 2, 4, 4))
        output = model(
            state,
            torch.zeros((1,), dtype=torch.long),
            cond=torch.zeros((1, 64)),
            spatial_condition=spatial_condition,
        )
        self.assertEqual(tuple(output.shape), tuple(state.shape))
        with self.assertRaisesRegex(ValueError, "spatial condition"):
            model(state, torch.zeros((1,), dtype=torch.long), cond=torch.zeros((1, 64)))

    def test_single_state_unet_has_six_inputs_and_one_output(self) -> None:
        model = Unet3D(
            dim=8,
            dim_mults=(1,),
            channels=1,
            spatial_condition_channels=5,
            cond_dim=64,
            attn_heads=1,
            attn_dim_head=8,
            resnet_groups=1,
        )
        self.assertEqual(model.init_conv.in_channels, 6)
        state = torch.zeros((1, 1, 2, 4, 4))
        output = model(
            state,
            torch.zeros((1,), dtype=torch.long),
            cond=torch.zeros((1, 64)),
            spatial_condition=torch.zeros((1, 5, 2, 4, 4)),
        )
        self.assertEqual(tuple(output.shape), tuple(state.shape))

    def test_lesion_only_state_and_soft_histogram_have_finite_gradients(self) -> None:
        diffusion = GaussianDiffusion_Nolatent(
            _ZeroDenoiser(),
            image_size=2,
            num_frames=1,
            spatial_shape=(1, 1, 2),
            channels=1,
            timesteps=4,
            data_type='gli',
            objective='pred_x0',
            gli_state_mode='lesion_only',
            hist_loss_weight=0.1,
            hist_loss_ramp_steps=500,
        )
        target = torch.tensor([[[[[0.75, -0.5]]]]], requires_grad=True)
        mask = torch.zeros((1, 4, 1, 1, 2))
        mask[:, 2, :, :, 0] = 1
        state = diffusion._gli_state_start(target, mask)
        self.assertEqual(float(state[0, 0, 0, 0, 1]), 0.0)

        prediction = torch.tensor([[[[[0.7, 0.0]]]]], requires_grad=True)
        hist = torch.zeros((1, 64))
        hist[:, 2 * 16 + 13] = 1.0
        details = soft_histogram_loss_details(prediction, mask, hist)
        self.assertTrue(torch.isfinite(details['loss']))
        details['loss'].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertAlmostEqual(diffusion._hist_weight_at_step(0), 0.0)
        self.assertAlmostEqual(diffusion._hist_weight_at_step(250), 0.05)
        self.assertAlmostEqual(diffusion._hist_weight_at_step(500), 0.1)

    def test_hydra_configs_are_complete_and_shapes_are_valid(self) -> None:
        config_dir = str(ROOT / "LeFusion" / "train" / "config")
        expected = {
            "gli_64x64x32": (32, 64, 64),
            "gli_80x96x80": (80, 80, 96),
            "gli_exp008_conditional_inpainting_64x64x32": (32, 64, 64),
            "gli_exp010_single_state_noise_64x64x32": (32, 64, 64),
            "gli_exp011_lesion_only_noise_64x64x32": (32, 64, 64),
            "gli_exp012_lesion_only_x0_hist_64x64x32": (32, 64, 64),
        }
        for config_name, shape in expected.items():
            with self.subTest(config_name=config_name):
                with initialize_config_dir(version_base=None, config_dir=config_dir):
                    cfg = compose(
                        config_name="base_cfg",
                        overrides=[f"+experiment={config_name}"],
                    )
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
                data_type, actual_shape = validate_training_config(cfg)
                self.assertEqual(data_type, "gli")
                self.assertEqual(actual_shape, shape)
                self.assertEqual(normalize_spatial_shape(shape), shape)


if __name__ == "__main__":
    unittest.main()
