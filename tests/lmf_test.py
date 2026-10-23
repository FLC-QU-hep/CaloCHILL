import unittest

import k_diffusion as K
import torch
from helpers import (
    SyntheticData,
    build_vae,
    first_batch,
    layer_kwargs,
    load_configs,
    make_dataset,
    make_loader,
)
from models.CaloHadronic import CaloHadronic_Attention, CaloHadronic_FM


class LatentFlowTest(SyntheticData, unittest.TestCase):
    """One training step of the latent model on a randomly initialised VAE (the VAE
    checkpoint training.py loads is not available in CI), then a sample decoded to
    a shower."""

    def run_latent_model(self, generative_model, model_cls, it):
        cfg_data, cfg_model = load_configs(
            "configs_LatentDiffusion.yaml", self.dataset_path
        )
        _, cfg_compressor = load_configs("configs_compressor.yaml", self.dataset_path)
        cfg_model.model.generative_model = generative_model
        cfg_model.latent.stats_batches = 2
        cfg_model.num_steps = 2  # EDM sampler steps, kept short

        # Train.get_model() for --net LatentDiff
        cfg_model.transformer.input_feats = int(
            cfg_compressor.vae.z_dim * cfg_compressor.vae.n_layers
        )
        cfg_model.downsample_factor = cfg_compressor.vae.p_dim_pions
        cfg_model.transformer.calorimeter_layers = cfg_data.calorimeter_layers
        vae = build_vae(cfg_data, cfg_compressor)
        vae.eval().requires_grad_(False)
        model = model_cls(cfg_model, "LatentDiff")

        loader = make_loader(make_dataset(cfg_data, cfg_model, "LatentDiff"))
        # main(): latent target 'mu' measures latent_scale / noise once at start-up
        model.measure_latent_stats(vae, loader, "cpu")
        self.assertEqual(int(model.latent_stats_ready), 1)
        self.assertTrue(float(model.latent_scale) > 0)

        # Train.forward() for --net LatentDiff
        batch = first_batch(loader)
        x = model.encode_latent_target(
            vae,
            batch["events"],
            ctx=batch["cond_feats"],
            y=batch["layer_cond"],
            mint_per_layer=batch["mint_per_layer"],
            padding=batch["padding_mask"],
        )
        noise = torch.randn_like(x)
        sample_density = K.config.make_sample_density(
            cfg_model.to_dict(["model_sigma"])["model_sigma"]
        )
        sigma = sample_density([x.shape[0]], device="cpu")
        loss = model.get_loss(
            x,
            noise,
            sigma,
            batch["cond_feats"],
            padding_mask=batch["padding_mask"],
            y=batch["layer_cond"],
            mint_per_layer=batch["mint_per_layer"],
            it=it,
            **layer_kwargs(batch),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.diffusion.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))

        # Train.val(): sample min(hits, p_dim) latent tokens, decode with the VAE
        model.eval()
        pad = batch["padding_mask"]
        n_tokens = min(int((~pad).sum(1).min()), cfg_compressor.vae.p_dim)
        z = model.sample(
            batch["cond_feats"],
            n_tokens,
            cfg_model,
            y=batch["layer_cond"],
            mint_per_layer=batch["mint_per_layer"],
            padding_mask=pad,
        )
        self.assertEqual(z.shape, (x.shape[0], n_tokens, x.shape[-1]))
        self.assertTrue(torch.isfinite(z).all())
        shower, _ = vae.sample(
            N=pad.shape[1],
            c=batch["cond_feats"],
            y=batch["layer_cond"],
            mint_per_layer=batch["mint_per_layer"],
            padding=pad,
            device="cpu",
            z=model.latent_to_decoder(z),
        )
        self.assertEqual(shower.shape, batch["events"].shape)
        self.assertTrue(torch.isfinite(shower).all())
        return model

    def test_mean_flow(self):
        # it=10 also runs the one-step diagnostics training logs
        model = self.run_latent_model("mean_flow", CaloHadronic_FM, it=10)
        self.assertTrue(model.mf_metrics)

    def test_edm(self):
        self.run_latent_model("edm", CaloHadronic_Attention, it=1)


if __name__ == "__main__":
    unittest.main()
