# UIE-DIS

轻量级水下图像增强一致性学生模型训练脚本。

## 数据组织

```text
data/paired_uie/
  train/
    input/
    target/
  val/
    input/
    target/
```

`input` 与 `target` 下的相对路径和文件名需要一一对应。

## 训练

```bash
pip install -r requirements.txt
python train.py --config configs/train_student.yaml
```

推荐先跑纯监督 baseline：

```bash
python train.py --config configs/train_supervised_baseline.yaml
```

当纯监督训练已经稳定超过输入基线后，再尝试 `configs/train_student.yaml` 这种带噪声/自一致的版本。

训练日志会输出：
- `train_loss`
- `val_loss`
- `val_psnr`
- `val_ssim`

训练过程中会保存：
- `latest.pth`
- `best.pth`（按 `val_loss`）
- `best_psnr.pth`
- `best_ssim.pth`

验证阶段默认保留原图分辨率计算指标；如需中心裁剪验证，可在 `configs/train_student.yaml` 里设置 `data.center_crop_eval: true`。

## 一步推理

单图或目录推理：

```bash
python infer.py \
  --checkpoint checkpoints/student/best.pth \
  --input data/paired_uie/val/input \
  --output results/val_pred
```

如果同时提供 GT 目录，会额外统计平均 `PSNR/SSIM`：

```bash
python infer.py \
  --checkpoint checkpoints/student/best.pth \
  --input data/paired_uie/val/input \
  --target data/paired_uie/val/target \
  --output results/val_pred
```
