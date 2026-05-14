import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from utils import (
    build_model,
    call_model,
    compute_batch_metrics,
    dump_json,
    extract_prediction,
    load_checkpoint,
    load_image_tensor,
    save_image_tensor,
)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='水下一致性蒸馏推理脚本')
    parser.add_argument('--input-path', required=True, help='单张图像或目录')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-spec', default=None)
    parser.add_argument('--model-kwargs-json', default='{}')
    parser.add_argument('--strict-load', action='store_true')
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--start-time', type=float, default=1.0)
    parser.add_argument('--end-time', type=float, default=0.0)
    parser.add_argument('--init-mode', default='input', choices=['input', 'input_noise'])
    parser.add_argument('--init-sigma', type=float, default=0.05)
    parser.add_argument('--target-dir', default=None, help='可选，用于推理后计算指标')
    parser.add_argument('--result-json', default=None)
    return parser.parse_args()


def list_inputs(path: str):
    input_path = Path(path)
    if input_path.is_file():
        return [input_path]
    files = [p for p in input_path.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort()
    return files


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.model_spec, json.loads(args.model_kwargs_json)).to(device)
    load_checkpoint(args.checkpoint, model, strict=args.strict_load, map_location='cpu')
    model.eval()

    input_files = list_inputs(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    with torch.no_grad():
        for image_path in tqdm(input_files, desc='Infer'):
            y = load_image_tensor(str(image_path), args.image_size).unsqueeze(0).to(device)
            if args.init_mode == 'input_noise':
                state = (y + torch.randn_like(y) * args.init_sigma).clamp(0.0, 1.0)
            else:
                state = y.clone()

            if args.steps <= 1:
                schedule = [args.start_time]
            else:
                schedule = torch.linspace(args.start_time, args.end_time, steps=args.steps).tolist()

            pred = state
            for time_value in schedule:
                t = torch.full((state.shape[0],), float(time_value), device=device)
                outputs = call_model(model, state, y, t)
                pred, _ = extract_prediction(outputs)
                pred = pred.clamp(0.0, 1.0)
                state = pred

            save_image_tensor(pred, str(output_dir / image_path.name))

            if args.target_dir:
                target_path = Path(args.target_dir) / image_path.name
                if target_path.exists():
                    target = load_image_tensor(str(target_path), args.image_size).unsqueeze(0).to(device)
                    metrics = compute_batch_metrics(pred, target)
                    metrics['file'] = image_path.name
                    all_metrics.append(metrics)

    if args.result_json:
        payload = {'files': all_metrics}
        if all_metrics:
            avg_metrics = {}
            for key in ['psnr', 'ssim', 'mae', 'uciqe', 'uiqm']:
                avg_metrics[key] = sum(item[key] for item in all_metrics) / len(all_metrics)
            payload['average'] = avg_metrics
        dump_json(payload, args.result_json)


if __name__ == '__main__':
    main()
