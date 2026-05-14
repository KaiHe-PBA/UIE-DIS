import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from tqdm import tqdm

from datasets import PairedImageDataset, build_dataloader
from losses import MultiTermLoss
from utils import (
    AverageMeter,
    build_model,
    call_model,
    dump_json,
    evaluate_image_pair,
    extract_prediction,
    load_checkpoint,
    move_batch_to_device,
    save_checkpoint,
    seed_everything,
)


def parse_json_argument(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    return json.loads(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='水下一致性蒸馏训练脚本')
    parser.add_argument('--train-input-dir', required=True)
    parser.add_argument('--train-target-dir', required=True)
    parser.add_argument('--val-input-dir', required=True)
    parser.add_argument('--val-target-dir', required=True)
    parser.add_argument('--save-dir', default='outputs/train')
    parser.add_argument('--model-spec', default=None, help='module:Class 或 /abs/path/file.py:Class')
    parser.add_argument('--model-kwargs-json', default='{}')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--strict-load', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--train-augment', action='store_true')
    parser.add_argument('--num-time-steps', type=int, default=1000)
    parser.add_argument('--sigma-min', type=float, default=0.0)
    parser.add_argument('--sigma-max', type=float, default=0.2)
    parser.add_argument('--time-sampling', default='uniform', choices=['uniform', 'quadratic'])
    parser.add_argument('--x-t-source', default='target', choices=['target', 'input'])
    parser.add_argument('--teacher-dir', default=None)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--consistency-weight', type=float, default=1.0)
    parser.add_argument('--reconstruction-weight', type=float, default=1.0)
    parser.add_argument('--ssim-weight', type=float, default=0.2)
    parser.add_argument('--color-weight', type=float, default=0.1)
    parser.add_argument('--edge-weight', type=float, default=0.1)
    parser.add_argument('--frequency-weight', type=float, default=0.05)
    parser.add_argument('--perceptual-weight', type=float, default=0.0)
    parser.add_argument('--reconstruction-mode', default='charbonnier', choices=['charbonnier', 'l1'])
    parser.add_argument('--consistency-mode', default='charbonnier', choices=['charbonnier', 'l1', 'l2'])
    return parser.parse_args()


@torch.no_grad()
def validate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    meters = {name: AverageMeter() for name in ['psnr', 'ssim', 'mae', 'uciqe', 'uiqm']}
    for batch in tqdm(loader, desc='Validate', leave=False):
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
            metric_values = evaluate_image_pair(
                pred[idx, :, :height, :width],
                batch['target'][idx, :, :height, :width],
            )
            for name, value in metric_values.items():
                meters[name].update(value, 1)
    return {name: meter.avg for name, meter in meters.items()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_enabled = args.amp and device.type == 'cuda'

    model = build_model(args.model_spec, parse_json_argument(args.model_kwargs_json)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    criterion = MultiTermLoss(
        consistency_weight=args.consistency_weight,
        reconstruction_weight=args.reconstruction_weight,
        ssim_weight=args.ssim_weight,
        color_weight=args.color_weight,
        edge_weight=args.edge_weight,
        frequency_weight=args.frequency_weight,
        perceptual_weight=args.perceptual_weight,
        reconstruction_mode=args.reconstruction_mode,
        consistency_mode=args.consistency_mode,
    ).to(device)

    train_dataset = PairedImageDataset(
        input_dir=args.train_input_dir,
        target_dir=args.train_target_dir,
        image_size=args.image_size,
        crop_size=args.crop_size,
        augment=args.train_augment,
        num_time_steps=args.num_time_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        time_sampling=args.time_sampling,
        x_t_source=args.x_t_source,
        teacher_dir=args.teacher_dir,
    )
    val_dataset = PairedImageDataset(
        input_dir=args.val_input_dir,
        target_dir=args.val_target_dir,
        image_size=args.image_size,
        crop_size=None,
        augment=False,
        num_time_steps=args.num_time_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        time_sampling='uniform',
        x_t_source=args.x_t_source,
        teacher_dir=args.teacher_dir,
    )
    train_loader = build_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = build_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    start_epoch = 0
    global_step = 0
    best_psnr = float('-inf')
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scaler=scaler,
            strict=args.strict_load,
            map_location='cpu',
        )
        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        global_step = int(checkpoint.get('step', 0))
        best_psnr = float(checkpoint.get('metrics', {}).get('psnr', float('-inf')))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    history = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_meter = AverageMeter()
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for batch_idx, batch in enumerate(progress):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = call_model(model, batch['x_t'], batch['input'], batch['t'], batch=batch)
                pred, extras = extract_prediction(outputs)
                pred = pred.clamp(0.0, 1.0)
                consistency_target = batch.get('teacher_target')
                if consistency_target is None:
                    consistency_target = extras.get('consistency_target') if isinstance(extras, dict) else None
                if consistency_target is None:
                    consistency_target = batch['target']
                total_loss, loss_items = criterion(pred, batch['target'], consistency_target)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = batch['target'].size(0)
            loss_meter.update(float(total_loss.item()), batch_size)
            global_step += 1

            if batch_idx % args.log_interval == 0:
                loss_text = ', '.join(f'{name}={value.item():.4f}' for name, value in loss_items.items())
                progress.set_postfix_str(loss_text)

        metrics = validate(model, val_loader, device)
        metrics['train_loss'] = loss_meter.avg
        history.append({'epoch': epoch, **metrics})
        dump_json({'history': history, 'config': config}, str(save_dir / 'train_history.json'))

        save_checkpoint(
            str(save_dir),
            'last',
            model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            step=global_step,
            metrics=metrics,
            config=config,
        )
        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            save_checkpoint(
                str(save_dir),
                'best',
                model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=global_step,
                metrics=metrics,
                config=config,
            )

        metric_text = ', '.join(f'{name}={value:.4f}' for name, value in metrics.items())
        print(f'[Epoch {epoch + 1}] {metric_text}')


if __name__ == '__main__':
    main()
