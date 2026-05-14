import argparse
import json

import torch
from tqdm import tqdm

from datasets import PairedImageDataset, build_dataloader
from utils import (
    AverageMeter,
    build_model,
    call_model,
    compute_batch_metrics,
    dump_json,
    extract_prediction,
    load_checkpoint,
    move_batch_to_device,
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
    return parser.parse_args()


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

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluate'):
            batch = move_batch_to_device(batch, device)
            outputs = call_model(model, batch['x_t'], batch['input'], batch['t'], batch=batch)
            pred, _ = extract_prediction(outputs)
            pred = pred.clamp(0.0, 1.0)
            values = compute_batch_metrics(pred, batch['target'])
            for name, value in values.items():
                meters[name].update(value, batch['target'].size(0))

    results = {name: meter.avg for name, meter in meters.items()}
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.result_json:
        dump_json(results, args.result_json)


if __name__ == '__main__':
    main()
