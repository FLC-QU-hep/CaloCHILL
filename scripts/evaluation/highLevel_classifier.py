import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.clas_utils import train


class DenseClassifier(nn.Module):
    def __init__(self, in_features, hidden_features, use_leaky_relu):
        super().__init__()

        if use_leaky_relu:
            Activation = nn.LeakyReLU
        else:
            Activation = nn.ReLU

        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            Activation(),
            nn.Linear(hidden_features, hidden_features),
            Activation(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def get_datasets(obs1, obs2):
    obs = torch.cat((obs1, obs2)).to(torch.get_default_dtype())
    num_samples = len(obs1)
    label1 = torch.zeros(num_samples, 1)
    label2 = torch.ones(num_samples, 1)
    label = torch.cat((label1, label2))

    r = torch.randperm(len(obs))
    obs = obs[r]
    label = label[r]
    to_split = [0.6, 0.2, 0.2]
    to_train, to_val = int(len(obs) * to_split[0]), int(len(obs) * to_split[1])
    split = [to_train, to_val, len(obs) - to_train - to_val]
    obs_train, obs_val, obs_test = torch.split(obs, split)
    label_train, label_val, label_test = torch.split(label, split)

    mean = torch.mean(obs_train, dim=0, keepdim=True)
    std = torch.std(obs_train, dim=0, unbiased=False, keepdim=True)
    std[std == 0] = (
        1  # always-empty profile bins: leave them at 0 instead of dividing by 0
    )
    obs_train -= mean
    obs_train /= std
    obs_val -= mean
    obs_val /= std
    obs_test -= mean
    obs_test /= std

    train_data = TensorDataset(obs_train, label_train)
    val_data = TensorDataset(obs_val, label_val)
    test_data = TensorDataset(obs_test, label_test)
    return train_data, val_data, test_data


def summarize(dirs):
    """Table X: per training (one figs/classifier/<name>/ each), the test AUC and OOD fractions at the epoch with the
    lowest validation loss; mean and std (ddof 0) over the trainings. Reads metrics.csv, or train.log (stdout) of older runs."""
    rows = []
    for d in dirs:
        if os.path.exists(d + "/metrics.csv"):
            m = np.loadtxt(d + "/metrics.csv", delimiter=",", skiprows=1, ndmin=2)[
                :, [2, 4, 5, 6]
            ]
        else:
            m = np.array(
                re.findall(
                    r"val loss: (\S+).*?=== test ===.*?AUC: (\S+).*?OOD Geant4: (\S+)\s*OOD model: (\S+)",
                    open(d + "/train.log").read(),
                    re.S,
                ),
                float,
            )
        rows.append(m[m[:, 0].argmin(), 1:])
        print(f"{d}: epoch {m[:, 0].argmin()} of {len(m)}, AUC {rows[-1][0]:.4f}")
    mu, sd = np.mean(rows, 0), np.std(rows, 0)
    print(
        f"AUC {mu[0]:.3f} +- {sd[0]:.3f}   OOD Geant4 [%] {100 * mu[1]:.2f} +- {100 * sd[1]:.2f}   "
        f"OOD model [%] {100 * mu[2]:.2f} +- {100 * sd[2]:.2f}   ({len(rows)} trainings)"
    )


def main():
    p = argparse.ArgumentParser(description="high-level classifier, Geant4 vs model")
    p.add_argument(
        "--summarize",
        nargs="+",
        metavar="DIR",
        help="only print the Table X row of the trainings in these figs/classifier/<name> folders",
    )
    p.add_argument(
        "--fake",
        help="comma-separated folders under files/projected_array/, pooled (required)",
    )
    p.add_argument(
        "--real",
        help="same for Geant4, e.g. Geant4/10GeV_g4win9.4-10.6_10000,... (required)",
    )
    p.add_argument("--name", default="high_level", help="figs/classifier/<name>/")
    p.add_argument("--hidden", type=int, default=4, help="hidden units per layer")
    p.add_argument(
        "--no-nhits",
        action="store_true",
        help="leave out every hit count: N hits of the 5 features",
    )
    p.add_argument(
        "--moments",
        action="store_true",
        help="also feed 4 profile moments: energy-weighted mean layer and its RMS, energy-weighted mean radius "
        "and its RMS in mm over the whole shower (ECal cells 5 mm, HCal 30 mm)",
    )
    args = p.parse_args()
    if args.summarize:
        return summarize(args.summarize)
    if not (args.fake and args.real):
        p.error("--fake and --real are required")
    dir_ = os.path.join(os.path.dirname(__file__), "../../files/projected_array/")
    names = ["e_sum_list", "occ_list", "cog_x", "cog_y", "cog_z"]
    if args.no_nhits:
        names.remove("occ_list")

    def npy(d, real, n):  # Geant4 files carry an '_r' suffix, except some cog ones
        return (
            d
            + "/"
            + n
            + ("_r" if real and os.path.exists(d + "/" + n + "_r.npy") else "")
            + ".npy"
        )

    def feats(d, real):
        """[N, features]: N hits, cog x/y/z and en_ratio = esum / inc_en (inc_en not a feature)."""

        def ld(n):
            return np.asarray(np.load(npy(d, real, n)), float).reshape(-1)

        feat = {n: ld(n) for n in names}
        feat["en_ratio"] = feat.pop("e_sum_list") / ld("inc_en")
        out = np.stack(list(feat.values()), axis=-1)
        if args.moments:
            out = np.concatenate([out, moments(d, real)], axis=1)
        return torch.tensor(out)

    def moments(d, real):
        """[N, 4]: energy-weighted mean layer and RMS, energy-weighted mean radius and RMS in mm (r is in cells of each
        section: 5 mm ECal, 30 mm HCal). From the per-hit arrays, shower after shower."""
        occ = np.load(npy(d, real, "occ_list")).astype(int)
        e = np.load(npy(d, real, "hits_list")).astype(float)
        layer = np.load(npy(d, real, "Y")).astype(float)
        r = np.asarray(
            np.load(npy(d, real, "e_radial"), mmap_mode="r")[0], float
        ) * np.where(layer < 30, 5.0, 30.0)
        n = len(occ)
        sid = np.repeat(np.arange(n), occ)
        w = lambda x: np.bincount(sid, weights=x, minlength=n)
        esum = np.maximum(w(e), 1e-12)
        cols = []
        for x in (layer, r):
            m = w(e * x) / esum
            cols += [m, np.sqrt(np.maximum(w(e * x**2) / esum - m**2, 0))]
        return np.stack(cols, axis=1)

    tensor_real = torch.cat(
        [feats(dir_ + d, True) for d in args.real.split(",")], dim=0
    )
    tensor_fake = torch.cat(
        [feats(dir_ + d, False) for d in args.fake.split(",")], dim=0
    )
    os.makedirs(
        os.path.join(os.path.dirname(__file__), "../../figs/classifier", args.name),
        exist_ok=True,
    )

    print("classifier type: high  level")
    model = DenseClassifier(
        in_features=tensor_fake.shape[1],
        hidden_features=args.hidden,
        use_leaky_relu=True,
    )

    print(model)
    print("")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        model,
        get_datasets(tensor_real, tensor_fake),
        args.name,
        device=device,
        lr=0.001,
        batch_size=32,
        epochs=100,
    )


if __name__ == "__main__":
    main()
