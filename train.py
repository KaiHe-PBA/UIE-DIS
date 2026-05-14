import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from uie_student.datasets import PairedUnderwaterDataset
from uie_student.losses import StudentLoss, StudentLossConfig
from uie_student.metrics import calc_batch_metrics
from uie_student.models.student import UWCNAFConsistencyStudent
from uie_student.utils import AverageMeter, ensure_dir, load_yaml, set_seed


def sigma_schedule(t: torch.Tensor, sigma_min: float = 0.002, sigma_max: float = 0.5) -> torch.Tensor:
    return sigma_min * (sigma_max / sigma_min) ** t


def sample_t(batch_size: int, device: torch.device, bias_to_zero_pow: float = 2.0, force_zero_prob: float = 0.1) -> torch.Tensor:
    t = torch.rand(batch_size, device=device) ** bias_to_zero_pow
    if force_zero_prob > 0:
        mask = torch.rand(batch_size, device=device) < force_zero_prob
        t = torch.where(mask, torch.zeros_like(t), t)
    return t


def make_xt(
    y: torch.Tensor,
    x_gt: torch.Tensor,
    t: torch.Tensor,
    p_center_y: float = 0.6,
    sigma_min: float = 0.002,
    sigma_max: float = 0.5,
) -> torch.Tensor:
    b = y.shape[0]
    eps = torch.randn_like(y)
    sigma = sigma_schedule(t, sigma_min=sigma_min, sigma_max=sigma_max).view(b, 1, 1, 1)
    use_y = (torch.rand(b, device=y.device) < p_center_y).view(b, 1, 1, 1)
    center = torch.where(use_y, y, x_gt)
    return center + sigma * eps


def get_epoch_sigma_max(epoch: int, cfg) -> float:
    warmup_epochs = cfg['train'].get('warmup_epochs', 0)
    noise_ramp_epochs = cfg['train'].get('noise_ramp_epochs', 0)
    target_sigma_max = cfg['train']['sigma_max']
    if epoch <= warmup_epochs:
        return 0.0
    if noise_ramp_epochs <= 0:
        return target_sigma_max
    progress = min(max(epoch - warmup_epochs, 0), noise_ramp_epochs) / float(noise_ramp_epochs)
    return target_sigma_max * progress


def build_dataloaders(cfg):
    augment_train = cfg['data'].get('augment_train', True)
    train_set = PairedUnderwaterDataset(
        data_root=cfg['data']['root'],
        split='train',
        patch_size=cfg['data']['patch_size'],
        augment=augment_train,
    )
    val_set = PairedUnderwaterDataset(
        data_root=cfg['data']['root'],
        split='val',
        patch_size=cfg['data']['patch_size'],
        augment=False,
        center_crop_eval=cfg['data'].get('center_crop_eval', False),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(cfg):
    return UWCNAFConsistencyStudent(
        base_dim=cfg['model']['base_dim'],
        enc_blocks=tuple(cfg['model']['enc_blocks']),
        dec_blocks=tuple(cfg['model']['dec_blocks']),
        time_emb_dim=cfg['model']['time_emb_dim'],
        time_mlp_dim=cfg['model']['time_mlp_dim'],
        use_residual_head=cfg['model']['use_residual_head'],
        use_gate_in_film=cfg['model']['use_gate_in_film'],
        residual_scale=cfg['model'].get('residual_scale', 0.1),
    )


def build_scheduler(optimizer, cfg):
    sched_cfg = cfg['train'].get('scheduler', {})
    if not sched_cfg.get('enabled', False):
        return None
    sched_type = sched_cfg.get('type', 'cosine')
    if sched_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg['train']['epochs'],
            eta_min=sched_cfg.get('min_lr', 1e-6),
        )
    raise ValueError(f"Unsupported scheduler type: {sched_type}")


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, cfg, epoch):
    model.train()
    meter = AverageMeter()
    log_interval = cfg['train']['log_interval']
    use_self_cons = epoch >= cfg['train']['self_consistency_start_epoch']
    epoch_sigma_max = get_epoch_sigma_max(epoch, cfg)

    for step, batch in enumerate(loader, start=1):
        y = batch['y'].to(device, non_blocking=True)
        x_gt = batch['x_gt'].to(device, non_blocking=True)
        b = y.shape[0]

        warmup_epochs = cfg['train'].get('warmup_epochs', 0)
        in_warmup = epoch <= warmup_epochs
        if in_warmup or epoch_sigma_max <= 0:
            t1 = torch.zeros(b, device=device)
            x_t1 = y
        else:
            t1 = sample_t(b, device, cfg['train']['t_bias_pow'], cfg['train']['force_zero_prob'])
            x_t1 = make_xt(y, x_gt, t1, cfg['train']['p_center_y'], cfg['train']['sigma_min'], epoch_sigma_max)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out1 = model(x_t1, y, t1, return_residual=True)
            losses = criterion(out1, x_gt)
            total = losses['loss_total']

            if (not in_warmup) and epoch_sigma_max > 0 and use_self_cons and cfg['train']['w_self_consistency'] > 0:
                t2 = sample_t(b, device, cfg['train']['t_bias_pow'], 0.0)
                x_t2 = make_xt(y, x_gt, t2, cfg['train']['p_center_y'], cfg['train']['sigma_min'], epoch_sigma_max)
                out2 = model(x_t2, y, t2, return_residual=False)
                pred1 = out1.get('x0_from_residual', out1['x0'])
                pred2 = out2.get('x0_from_residual', out2['x0'])
                loss_cons = F.mse_loss(pred1, pred2.detach())
                losses['loss_self_consistency'] = loss_cons
                total = total + cfg['train']['w_self_consistency'] * loss_cons

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            optimizer.step()

        meter.update(float(total.detach().cpu()), b)

        if step % log_interval == 0:
            msg = f"epoch={epoch} step={step}/{len(loader)} loss={meter.avg:.6f}"
            if 'loss_self_consistency' in losses:
                msg += f" cons={float(losses['loss_self_consistency'].detach().cpu()):.6f}"
            msg += f" sigma_max={epoch_sigma_max:.4f}"
            pred_stats = out1.get('x0_from_residual', out1['x0']).detach()
            msg += (
                f" pred_mean={float(pred_stats.mean().cpu()):.4f}"
                f" gt_mean={float(x_gt.mean().detach().cpu()):.4f}"
                f" pred_min={float(pred_stats.amin().cpu()):.3f}"
                f" pred_max={float(pred_stats.amax().cpu()):.3f}"
            )
            print(msg)

    return meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    input_psnr_meter = AverageMeter()
    input_ssim_meter = AverageMeter()
    for batch in loader:
        y = batch['y'].to(device, non_blocking=True)
        x_gt = batch['x_gt'].to(device, non_blocking=True)
        t = torch.zeros(y.shape[0], device=device)
        out = model(y, y, t, return_residual=True)
        pred = out.get('x0_from_residual', out['x0']).clamp(0.0, 1.0)
        losses = criterion({'x0': pred}, x_gt)
        metrics = calc_batch_metrics(pred, x_gt)
        input_metrics = calc_batch_metrics(y, x_gt)
        batch_size = y.shape[0]
        loss_meter.update(float(losses['loss_total'].detach().cpu()), batch_size)
        psnr_meter.update(float(metrics['psnr'].mean().detach().cpu()), batch_size)
        ssim_meter.update(float(metrics['ssim'].mean().detach().cpu()), batch_size)
        input_psnr_meter.update(float(input_metrics['psnr'].mean().detach().cpu()), batch_size)
        input_ssim_meter.update(float(input_metrics['ssim'].mean().detach().cpu()), batch_size)
    return {
        'loss': loss_meter.avg,
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg,
        'input_psnr': input_psnr_meter.avg,
        'input_ssim': input_ssim_meter.avg,
    }


def save_checkpoint(model, optimizer, epoch, best_val, cfg, tag):
    save_dir = Path(cfg['train']['save_dir'])
    ensure_dir(str(save_dir))
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_val': best_val,
        'config': cfg,
    }
    torch.save(ckpt, save_dir / f'{tag}.pth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_student.yaml')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg['train']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['train']['amp'] and device.type == 'cuda')
    scaler = scaler if scaler.is_enabled() else None

    criterion = StudentLoss(StudentLossConfig(
        w_recon=cfg['loss']['w_recon'],
        w_ssim=cfg['loss']['w_ssim'],
        w_color=cfg['loss']['w_color'],
        w_edge=cfg['loss']['w_edge'],
        w_freq=cfg['loss']['w_freq'],
        use_charbonnier=cfg['loss']['use_charbonnier'],
    )).to(device)

    best_val = float('inf')
    best_psnr = float('-inf')
    best_ssim = float('-inf')
    for epoch in range(1, cfg['train']['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, cfg, epoch)
        val_metrics = validate(model, val_loader, criterion, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_psnr={val_metrics['psnr']:.4f} "
            f"val_ssim={val_metrics['ssim']:.4f} "
            f"input_psnr={val_metrics['input_psnr']:.4f} "
            f"input_ssim={val_metrics['input_ssim']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6e} "
            f"sigma_max={get_epoch_sigma_max(epoch, cfg):.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        save_checkpoint(model, optimizer, epoch, best_val, cfg, 'latest')
        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            save_checkpoint(model, optimizer, epoch, best_val, cfg, 'best')
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            save_checkpoint(model, optimizer, epoch, best_psnr, cfg, 'best_psnr')
        if val_metrics['ssim'] > best_ssim:
            best_ssim = val_metrics['ssim']
            save_checkpoint(model, optimizer, epoch, best_ssim, cfg, 'best_ssim')


if __name__ == '__main__':
    main()
