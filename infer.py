import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from train import build_model
from uie_student.metrics import calc_batch_metrics
from uie_student.utils import ensure_dir


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return TF.to_tensor(image).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    ensure_dir(str(path.parent))
    image = TF.to_pil_image(tensor.squeeze(0).cpu().clamp(0.0, 1.0))
    image.save(path)


def resolve_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / input_path.name
    return output_root / input_path.relative_to(input_root)


def resolve_target_path(image_path: Path, input_root: Path, target_root: Path) -> Path:
    if input_root.is_file():
        if target_root.is_file():
            return target_root
        return target_root / image_path.name
    return target_root / image_path.relative_to(input_root)


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config")
    if cfg is None:
        raise RuntimeError("Checkpoint does not contain training config, cannot rebuild model.")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model,
    input_path: Path,
    output_dir: Path,
    target_root: Optional[Path],
    device: torch.device,
) -> Tuple[int, float, float]:
    input_paths = list_images(input_path)
    if not input_paths:
        raise RuntimeError(f"No images found under {input_path}")

    total_psnr = 0.0
    total_ssim = 0.0
    metric_count = 0

    for image_path in input_paths:
        y = load_image(image_path).to(device)
        t = torch.zeros(1, device=device)
        out = model(y, y, t, return_residual=True)
        pred = out.get("x0_from_residual", out["x0"]).clamp(0.0, 1.0)

        save_path = resolve_output_path(image_path, input_path, output_dir)
        save_image(pred, save_path)

        if target_root is not None:
            target_path = resolve_target_path(image_path, input_path, target_root)
            if target_path.exists():
                target = load_image(target_path).to(device)
                metrics = calc_batch_metrics(pred, target)
                psnr = float(metrics["psnr"].mean().cpu())
                ssim = float(metrics["ssim"].mean().cpu())
                total_psnr += psnr
                total_ssim += ssim
                metric_count += 1
                print(f"{image_path.name}: PSNR={psnr:.4f}, SSIM={ssim:.4f}")
            else:
                print(f"[warn] missing target for {image_path}: {target_path}")
        else:
            print(f"saved: {save_path}")

    return metric_count, total_psnr, total_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="single image or folder")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target", type=str, default=None, help="optional target folder for PSNR/SSIM")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    ensure_dir(str(output_dir))
    target_root = Path(args.target) if args.target is not None else None

    metric_count, total_psnr, total_ssim = run_inference(
        model=model,
        input_path=input_path,
        output_dir=output_dir,
        target_root=target_root,
        device=device,
    )

    if metric_count > 0:
        print(f"avg_psnr={total_psnr / metric_count:.4f}")
        print(f"avg_ssim={total_ssim / metric_count:.4f}")


if __name__ == "__main__":
    main()
