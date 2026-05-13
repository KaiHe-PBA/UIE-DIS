import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from train import build_model
from uie_student.datasets import PairedUnderwaterDataset
from uie_student.losses import StudentLoss, StudentLossConfig
from uie_student.metrics import calc_batch_metrics
from uie_student.utils import load_yaml, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_student.yaml")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PairedUnderwaterDataset(
        data_root=cfg["data"]["root"],
        split="train",
        patch_size=cfg["data"]["patch_size"],
        augment=False,
        center_crop_eval=True,
    )
    subset = Subset(dataset, list(range(min(args.num_samples, len(dataset)))))
    loader = DataLoader(subset, batch_size=min(args.num_samples, 4), shuffle=True, num_workers=0)

    model = build_model(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
    criterion = StudentLoss(StudentLossConfig(w_recon=1.0, w_ssim=0.2, w_color=0.0, w_edge=0.0, w_freq=0.0, use_charbonnier=True)).to(device)

    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        y = batch["y"].to(device)
        x_gt = batch["x_gt"].to(device)
        t = torch.zeros(y.shape[0], device=device)

        optimizer.zero_grad(set_to_none=True)
        out = model(y, y, t, return_residual=True)
        losses = criterion(out, x_gt)
        losses["loss_total"].backward()
        optimizer.step()

        if step % 20 == 0:
            pred = out.get("x0_from_residual", out["x0"]).detach().clamp(0.0, 1.0)
            metrics = calc_batch_metrics(pred, x_gt)
            print(
                f"step={step} loss={float(losses['loss_total'].detach().cpu()):.6f} "
                f"psnr={float(metrics['psnr'].mean().detach().cpu()):.4f} "
                f"ssim={float(metrics['ssim'].mean().detach().cpu()):.4f}"
            )


if __name__ == "__main__":
    main()
