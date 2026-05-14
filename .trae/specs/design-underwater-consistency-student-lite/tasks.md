# Tasks
- [x] Task 1: 明确学生模型总体拓扑与复杂度预算，确定 Restormer-lite 风格 encoder-decoder、condition encoder、time embedding 和输出头的整体连接关系。
  - [x] SubTask 1.1: 固定默认输入输出接口，包括 `x_t`、`y`、`t` 与 `x_0_pred`
  - [x] SubTask 1.2: 明确 4-stage encoder、latent、4-stage decoder 的通道数、分辨率和 block 数
  - [x] SubTask 1.3: 给出默认轻量化预算，包括 attention heads、FFN 扩展倍率和上/下采样方式

- [x] Task 2: 设计多尺度 `condition encoder` 与逐尺度条件注入机制，确保其兼顾轻量化、高分辨率恢复和少步一致性蒸馏稳定性。
  - [x] SubTask 2.1: 定义 `condition encoder` 的多尺度输出与主干尺度对齐规则
  - [x] SubTask 2.2: 确定默认融合机制为 gated residual fusion，并明确其在每个 stage 中的位置
  - [x] SubTask 2.3: 给出可选 cross-attention 增强路径及其启用条件

- [x] Task 3: 设计 time embedding 注入与 stage 内部模块结构，明确 attention、FFN、skip connection 和条件融合的先后顺序。
  - [x] SubTask 3.1: 定义 time embedding 的生成方式和维度
  - [x] SubTask 3.2: 明确 time embedding 在 encoder、latent、decoder 各 stage 的调制形式
  - [x] SubTask 3.3: 写清每个 block 的 forward 逻辑和残差连接路径

- [x] Task 4: 制定可直接落地的 PyTorch 实现规格，要求后续实现时能直接写出训练和测试代码。
  - [x] SubTask 4.1: 明确模块划分，包括 `PatchEmbed`、`LiteTransformerBlock`、`CondEncoder`、`FusionBlock`、`TimeMLP`、`OutputHead`
  - [x] SubTask 4.2: 为每个 stage 给出输入输出张量尺寸示例，默认基于 256x256 输入
  - [x] SubTask 4.3: 明确 forward 数据流、输出字典与可选辅助头接口

- [x] Task 5: 设计完整训练损失与推理策略，覆盖一致性蒸馏、水下恢复质量和工程稳定性。
  - [x] SubTask 5.1: 明确 consistency loss、重建损失、SSIM/结构损失、perceptual loss、颜色校正损失的定义与权重建议
  - [x] SubTask 5.2: 评估是否加入频域、边缘保持、measurement consistency、degradation consistency 等可选约束
  - [x] SubTask 5.3: 定义 1~4 步推理策略、`x_0` 预测使用方式以及数值稳定建议

- [x] Task 6: 补充工程建议与验证要点，确保后续实现兼顾显存、速度和效果。
  - [x] SubTask 6.1: 说明默认配置在显存、推理速度和恢复效果上的取舍
  - [x] SubTask 6.2: 给出可进一步轻量化但不明显伤害性能的改造点
  - [x] SubTask 6.3: 明确后续实现后的单元验证、形状验证和训练稳定性检查项

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 1
- Task 4 depends on Task 1, Task 2, Task 3
- Task 5 depends on Task 1, Task 4
- Task 6 depends on Task 4, Task 5
