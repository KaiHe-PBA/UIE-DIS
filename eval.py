from __future__ import annotations

import argparse
from pathlib import Path

from uie_dis.data import UnderwaterImagePairDataset, build_dataloader
from uie_dis.eval_tools import load_model_from_checkpoint, run_paired_evaluation
from uie_dis.training import resolve_device


def parse_bool(value: str) -> bool:
    normalized = value.lower().strip()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on paired validation set.")
    parser.add_argument("--checkpoint", required=True, help="Path to training checkpoint.")
    parser.add_argument("--val-input-dir", default="/data/val/input")
    parser.add_argument("--val-target-dir", default="/data/val/target")
    parser.add_argument("--output-dir", default="/workspace/eval_outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--image-ext", default="png", help="Export extension, e.g. png or ppm.")
    parser.add_argument("--use-ema", type=parse_bool, default=True)
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument("--export-images", type=parse_bool, default=True)
    parser.add_argument("--save-input", type=parse_bool, default=False)
    parser.add_argument("--save-target", type=parse_bool, default=False)
    parser.add_argument("--save-history", type=parse_bool, default=False)
    parser.add_argument("--clamp-min", type=float, default=0.0)
    parser.add_argument("--clamp-max", type=float, default=1.0)
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    device = resolve_device(args.device)
    model, metadata = load_model_from_checkpoint(args.checkpoint, device=device, use_ema=args.use_ema)
    dataset = UnderwaterImagePairDataset(
        args.val_input_dir,
        args.val_target_dir,
        patch_size=None,
        training=False,
        random_flip=False,
        center_crop_validation=False,
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    summary = run_paired_evaluation(
        model=model,
        dataloader=dataloader,
        output_dir=args.output_dir,
        device=device,
        num_steps=args.num_steps,
        image_ext=args.image_ext,
        amp=args.amp,
        export_images=args.export_images,
        save_input=args.save_input,
        save_target=args.save_target,
        save_history=args.save_history,
        clamp_range=(args.clamp_min, args.clamp_max),
    )
    print("Evaluation completed.")
    print(f"Loaded checkpoint state: {metadata['loaded_state']}")
    print(
        f"PSNR={summary['metrics']['psnr']:.3f}, "
        f"SSIM={summary['metrics']['ssim']:.4f}, "
        f"L1={summary['metrics']['l1']:.5f}"
    )
    print(f"Saved outputs to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
