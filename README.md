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

## 一步推理

```python
import torch
from uie_student.models.student import UWCNAFConsistencyStudent

ckpt = torch.load('checkpoints/student/best.pth', map_location='cpu')
model = UWCNAFConsistencyStudent()
model.load_state_dict(ckpt['model'])
model.eval()

y = torch.rand(1, 3, 256, 256)
t = torch.zeros(1)
with torch.no_grad():
    out = model(y, y, t, return_residual=True)
    pred = out.get('x0_from_residual', out['x0']).clamp(0, 1)
```
