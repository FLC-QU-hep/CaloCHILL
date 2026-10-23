import unittest

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


class VAETest(SyntheticData, unittest.TestCase):
    def test_loss_backward_and_reco(self):
        cfg_data, cfg_model = load_configs("configs_compressor.yaml", self.dataset_path)
        vae = build_vae(cfg_data, cfg_model)
        batch = first_batch(
            make_loader(make_dataset(cfg_data, cfg_model, "Compressor"))
        )
        events = batch["events"]

        # the get_loss call of Train.forward (net Compressor), beta at iteration 1
        loss_dict = vae.get_loss(
            events.clone(),
            ctx=batch["cond_feats"],
            y=batch["layer_cond"],
            mint_per_layer=batch["mint_per_layer"],
            padding_mask=batch["padding_mask"],
            beta=1e-4,
            it=1,
            **layer_kwargs(batch),
        )
        self.assertIsNotNone(loss_dict, "output and target shapes differ")
        for key in ("loss", "rec_loss", "kl_loss"):
            self.assertTrue(torch.isfinite(loss_dict[key]), key)
        loss_dict["loss"].backward()
        grads = [p.grad for p in vae.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads if g is not None))

        # the latent the LMF trains on: effective_p_dim tokens of z_dim
        mu, logvar = vae.encode_for_latent(
            events,
            ctx=batch["cond_feats"],
            y=batch["layer_cond"],
            mint_per_layer=batch["mint_per_layer"],
            padding=batch["padding_mask"],
            return_params=True,
        )
        p_dim = min(int((~batch["padding_mask"]).sum(1).min()), cfg_model.vae.p_dim)
        self.assertEqual(vae.effective_p_dim, p_dim)
        self.assertEqual(mu.shape[0], events.shape[0])
        self.assertEqual(mu.shape[-1], cfg_model.vae.z_dim)
        self.assertGreaterEqual(mu.shape[1], p_dim)
        self.assertTrue(torch.isfinite(mu).all() and torch.isfinite(logvar).all())

        rec, _ = vae.reco(
            events,
            c=batch["cond_feats"],
            y=batch["layer_cond"],
            mint_per_layer=batch["mint_per_layer"],
            padding=batch["padding_mask"],
        )
        self.assertEqual(rec.shape, events.shape)
        self.assertTrue(torch.isfinite(rec).all())


if __name__ == "__main__":
    unittest.main()
