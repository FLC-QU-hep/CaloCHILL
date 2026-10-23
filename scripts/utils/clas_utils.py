import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_metrics(
    all_scores, all_label, train_loss=None, val_loss=None, test_loss=None, name=None
):
    auc = roc_auc_score(
        all_label.cpu().flatten().numpy(), all_scores.cpu().flatten().numpy()
    )
    acc = (
        ((all_scores > 0.5).float() == all_label).float().mean().item()
    )  # torchmetrics Accuracy(task="binary")
    jsd = calc_JSD(all_scores, all_label).item()
    jsd2 = calc_JSD(all_scores).item()

    if name:
        print(f"=== {name} ===")
    if train_loss:
        print("train loss:", train_loss)
    if val_loss:
        print("val loss:", val_loss)
    if test_loss:
        print("test loss:", test_loss)
    print("accuracy:", acc)
    print("AUC:", auc)
    print("JSD:", jsd)
    print("JSD w/o labels:", jsd2)
    ood = (all_scores <= 0.01) | (
        all_scores >= 0.99
    )  # OOD fraction: classifier > 99% sure, per class (0 = Geant4, 1 = model)
    ood_g4, ood_model = (
        ood[all_label == 0].float().mean().item(),
        ood[all_label == 1].float().mean().item(),
    )
    print("OOD Geant4:", ood_g4)
    print("OOD model:", ood_model)
    print("")
    sys.stdout.flush()
    return auc, ood_g4, ood_model


def calc_JSD(ratio, label=None):
    if label is None:
        label = ratio
    d1 = torch.mean(label * torch.log2(ratio + 1e-8))
    d2 = torch.mean((1 - label) * torch.log2(1.0 - ratio + 1e-8))
    return d1 + d2 + 1.0


def plotting_losses(train_loss, val_loss, test_loss, dir_name=""):
    plt.figure(1)
    plt.plot(val_loss, label="val")
    plt.plot(test_loss, label="test")
    plt.plot(train_loss, label="train")
    plt.legend()
    plt.savefig("./figs/classifier/" + dir_name + "/loss.png")
    plt.close()


def train_epoch(model, data, device, optimizer, criterion):
    """Batches are (*model inputs, label)."""
    model.train()
    average_loss = 0
    for *inputs, label in data:
        inputs = [x.to(device) for x in inputs]
        label = label.to(device)

        optimizer.zero_grad()
        scores = model(*inputs)
        loss = criterion(scores, label)
        loss.backward()
        optimizer.step()

        average_loss += len(inputs[0]) * loss.item()
    average_loss /= len(data.dataset)
    return average_loss


@torch.no_grad()
def validation(model, data, device, criterion):
    model.eval()

    val_loss = 0
    all_label = []
    all_scores = []

    for *inputs, label in data:
        inputs = [x.to(device) for x in inputs]
        label = label.to(device)

        scores = model(*inputs)
        loss = criterion(scores, label)

        val_loss += len(inputs[0]) * loss.item()
        all_label.append(label)
        all_scores.append(torch.sigmoid(scores))

    all_label = torch.cat(all_label)
    all_scores = torch.cat(all_scores)
    val_loss /= len(data.dataset)

    return all_scores, all_label, val_loss


def train(model, datasets, dir_name, lr=1e-3, batch_size=256, epochs=100, device="cpu"):
    """datasets: (train, val, test); figures and metrics.csv (one row per epoch) go to figs/classifier/<dir_name>/."""
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    train_loader = DataLoader(
        datasets[0],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=32,
    )
    val_loader, test_loader = (
        DataLoader(d, batch_size=2**11, shuffle=False, pin_memory=True, num_workers=32)
        for d in datasets[1:]
    )
    to_numpy = lambda x: x.to(dtype=torch.float64, device="cpu").flatten().numpy()  # noqa: E731

    tr_loss, vl_loss, t_loss = [], [], []
    with open(f"./figs/classifier/{dir_name}/metrics.csv", "w") as fh:
        fh.write(
            "epoch,train_loss,val_loss,test_loss,test_auc,test_ood_g4,test_ood_model\n"
        )
    for i in tqdm(range(epochs)):
        train_loss = train_epoch(model, train_loader, device, optimizer, criterion)
        scores, labels, val_loss = validation(model, val_loader, device, criterion)
        evaluate_metrics(
            scores,
            labels,
            train_loss=train_loss,
            val_loss=val_loss,
            name=f"epoch {i:3d}",
        )
        calibrator = IsotonicRegression(
            out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6
        )
        calibrator.fit(to_numpy(scores), to_numpy(labels))
        scores_test, labels_test, test_loss = validation(
            model, test_loader, device, criterion
        )
        rescaled_scores = (
            torch.from_numpy(calibrator.predict(to_numpy(scores_test)))
            .to(dtype=torch.float32, device=device)
            .unflatten(0, (-1, 1))
        )
        test_metrics = evaluate_metrics(
            rescaled_scores, labels_test, test_loss=test_loss, name="test"
        )
        with open(f"./figs/classifier/{dir_name}/metrics.csv", "a") as fh:
            fh.write(
                ",".join(map(str, (i, train_loss, val_loss, test_loss, *test_metrics)))
                + "\n"
            )
        tr_loss.append(train_loss)
        vl_loss.append(val_loss)
        t_loss.append(test_loss)
        plotting_losses(tr_loss, vl_loss, t_loss, dir_name=dir_name)
