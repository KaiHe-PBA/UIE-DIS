import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from datasets import PairedImageDataset, build_dataloader
from utils import (
    AverageMeter,
    build_model,
    call_model,
    dump_json,
    evaluate_image_pair,
    extract_prediction,
    load_checkpoint,
    move_batch_to_device,
    save_image_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='水下增强评估脚本')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--target-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-spec', default=None)
    parser.add_argument('--model-kwargs-json', default='{}')
    parser.add_argument('--strict-load', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--num-time-steps', type=int, default=1000)
    parser.add_argument('--sigma-min', type=float, default=0.0)
    parser.add_argument('--sigma-max', type=float, default=0.2)
    parser.add_argument('--x-t-source', default='target', choices=['target', 'input'])
    parser.add_argument('--result-json', default=None)
    parser.add_argument('--save-image-dir', default=None, help='评估时保存增强结果图像的目录')
    return parser.parse_args()


def build_output_path(input_path: str, input_root: str, output_root: Path) -> Path:
    source_path = Path(input_path)
    root_path = Path(input_root)
    try:
        relative_path = source_path.relative_to(root_path)
    except ValueError:
        relative_path = Path(source_path.name)
    return output_root / relative_path


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.model_spec, json.loads(args.model_kwargs_json)).to(device)
    load_checkpoint(args.checkpoint, model, strict=args.strict_load, map_location='cpu')
    model.eval()

    dataset = PairedImageDataset(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        image_size=args.image_size,
        crop_size=None,
        augment=False,
        num_time_steps=args.num_time_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        x_t_source=args.x_t_source,
    )
    loader = build_dataloader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    meters = {name: AverageMeter() for name in ['psnr', 'ssim', 'mae', 'uciqe', 'uiqm']}
    save_image_root = Path(args.save_image_dir) if args.save_image_dir else None

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluate'):
            batch = move_batch_to_device(batch, device)
            outputs = call_model(model, batch['x_t'], batch['input'], batch['t'], batch=batch)
            pred, _ = extract_prediction(outputs)
            pred = pred.clamp(0.0, 1.0)
            original_hw = batch.get('original_hw')
            batch_size = batch['target'].size(0)
            for idx in range(batch_size):
                if original_hw is not None:
                    height = int(original_hw[idx, 0].item())
                    width = int(original_hw[idx, 1].item())
                else:
                    height = batch['target'].shape[-2]
                    width = batch['target'].shape[-1]
                values = evaluate_image_pair(
                    pred[idx, :, :height, :width],
                    batch['target'][idx, :, :height, :width],
                )
                for name, value in values.items():
                    meters[name].update(value, 1)
                if save_image_root is not None:
                    input_paths = batch.get('input_path')
                    input_path = input_paths[idx] if isinstance(input_paths, list) else f'eval_{idx:05d}.png'
                    output_path = build_output_path(input_path, args.input_dir, save_image_root)
                    save_image_tensor(pred[idx, :, :height, :width], str(output_path))

    results = {name: meter.avg for name, meter in meters.items()}
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.result_json:
        dump_json(results, args.result_json)


if __name__ == '__main__':
    main()
