import unittest

import k_diffusion as K
import torch
from helpers import SyntheticData, first_batch, load_configs, make_dataset, make_loader
from models.CaloHadronic_legacy import CaloClouds2_Attention


class CaloHadronicLegacyTest(SyntheticData, unittest.TestCase):
    def test_loss_backward_and_heun_sample(self):
        cfg_data, cfg_model = load_configs(
            "configs_CaloHadronic.yaml", self.dataset_path
        )
        # Train.get_model() for --net CaloHadronic
        cfg_model.transformer.calorimeter_layers = cfg_data.calorimeter_layers
        model = CaloClouds2_Attention(cfg_model)
        batch = first_batch(
            make_loader(make_dataset(cfg_data, cfg_model, "CaloHadronic"))
        )

        # Train.forward() for --net CaloHadronic
        x = batch["events"]
        sample_density = K.config.make_sample_density(
            cfg_model.to_dict(["model_sigma"])["model_sigma"]
        )
        loss = model.get_loss(
            x,
            torch.randn_like(x),
            sample_density([x.shape[0]], device="cpu"),
            batch["cond_feats"],
            padding_mask=batch["padding_mask"],
            y=batch["layer_cond"],
            mint_per_layer=batch["mint_per_layer"],
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

        # Train.val(): one generated point per hit slot, heun with few steps
        cfg_model.num_steps = 3
        model.eval()
        sample = model.sample(
            batch["cond_feats"],
            batch["layer_cond"].shape[1],
            cfg_model,
            y=batch["layer_cond"],
            mint_per_layer=batch["mint_per_layer"],
            padding_mask=batch["padding_mask"],
        )
        self.assertEqual(sample.shape, x.shape)
        self.assertTrue(torch.isfinite(sample).all())


if __name__ == "__main__":
    unittest.main()
