from __future__ import annotations

import argparse
from pathlib import Path

from uie_dis.data import build_dataloader
from uie_dis.eval_tools import ImageDirectoryDataset, load_model_from_checkpoint, run_unpaired_inference
from uie_dis.training import resolve_device


def parse_bool(value: str) -> bool:
    normalized = value.lower().strip()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multistep inference and export enhanced images.")
    parser.add_argument("--checkpoint", required=True, help="Path to training checkpoint.")
    parser.add_argument("--input-dir", default="/data/val/input", help="Directory containing input underwater images.")
    parser.add_argument("--output-dir", default="/workspace/inference_outputs", help="Directory to save outputs.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--image-ext", default="png", help="Export extension, e.g. png or ppm.")
    parser.add_argument("--use-ema", type=parse_bool, default=True)
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument("--save-input", type=parse_bool, default=False)
    parser.add_argument("--save-history", type=parse_bool, default=False)
    parser.add_argument("--clamp-min", type=float, default=0.0)
    parser.add_argument("--clamp-max", type=float, default=1.0)
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    device = resolve_device(args.device)
    model, metadata = load_model_from_checkpoint(args.checkpoint, device=device, use_ema=args.use_ema)
    dataset = ImageDirectoryDataset(args.input_dir)
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    summary = run_unpaired_inference(
        model=model,
        dataloader=dataloader,
        output_dir=args.output_dir,
        device=device,
        num_steps=args.num_steps,
        image_ext=args.image_ext,
        amp=args.amp,
        save_input=args.save_input,
        save_history=args.save_history,
        clamp_range=(args.clamp_min, args.clamp_max),
    )
    print("Inference completed.")
    print(f"Loaded checkpoint state: {metadata['loaded_state']}")
    print(f"Saved {summary['num_samples']} enhanced images to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
