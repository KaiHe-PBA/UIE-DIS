# 水下图像增强一致性蒸馏轻量学生模型 Spec

## Why
当前面向水下图像增强的一致性蒸馏学生模型，往往在轻量化后显著削弱全局建模与高分辨率恢复能力，导致 1~4 步推理下的细节、颜色和结构一致性不足。本变更旨在设计一个更强一些但仍受控复杂度的 Restormer-lite 风格学生网络，兼顾长程依赖建模、条件引导与少步一致性蒸馏稳定性。

## What Changes
- 新增一个面向一致性蒸馏的条件式水下图像增强学生网络规格，主干采用 Restormer-lite 风格 encoder-decoder。
- 明确双输入接口：`x_t` 作为当前一致性状态，`y` 作为原始水下图像条件。
- 新增多尺度 `condition encoder` 规格，用于从 `y` 中提取跨尺度上下文条件特征。
- 新增逐尺度条件注入机制，优先采用轻量 gated residual fusion，并保留 cross-attention 作为可选增强分支。
- 新增全局 time embedding 注入方案，要求每个 stage 使用同一时间语义并保持输出空间一致。
- 新增 `x_0` 预测头与可选颜色校正辅助头规格，支持稳定的一致性蒸馏训练。
- 新增完整训练损失规格，包括 consistency、重建、结构、感知、颜色与可选频域/边缘/退化一致性约束。
- 新增高分辨率推理与工程落地约束，包括通道宽度、block 数、attention heads、FFN 压缩和显存控制策略。

## Impact
- Affected specs: 水下图像增强学生模型、条件恢复网络、多步到少步一致性蒸馏训练、轻量高分辨率恢复
- Affected code: `models/` 下学生模型定义、`losses/` 下训练损失、`train/` 或 `engine/` 下训练逻辑、`configs/` 下网络与损失配置、`infer/` 下少步推理接口

## ADDED Requirements
### Requirement: Restormer-lite 条件学生主干
系统 SHALL 提供一个 Restormer-lite 风格的 encoder-decoder 学生模型，以 `x_t` 和 `y` 为联合输入，支持 256x256 默认训练并可扩展到更高分辨率。

#### Scenario: 基础主干结构
- **WHEN** 构建学生模型主干
- **THEN** 模型包含 `patch embed -> 4-stage encoder -> latent -> 4-stage decoder -> 输出头`
- **AND** encoder 与 decoder 采用对称多尺度结构
- **AND** 每个 stage 明确给出输入输出通道数、block 数与上下采样规则
- **AND** 默认配置需显著轻于标准 Restormer

#### Scenario: 轻量化约束
- **WHEN** 选择学生模型配置
- **THEN** 默认版本应采用较小通道宽度、较少 blocks、较少 attention heads 和更轻的 FFN 扩展倍率
- **AND** 保留 Restormer 核心高效 transformer 归纳偏置
- **AND** 推理复杂度适配 1~4 步一致性蒸馏推理

### Requirement: 多尺度条件编码与注入
系统 SHALL 提供独立的 `condition encoder`，从条件图像 `y` 中提取与主干各尺度对齐的多尺度上下文特征，并在主干各 stage 进行融合。

#### Scenario: 条件编码
- **WHEN** 输入原始水下图像 `y`
- **THEN** `condition encoder` 输出与 backbone 各尺度空间分辨率匹配的条件特征
- **AND** 条件特征包含浅层颜色/雾化信息与深层结构/上下文信息

#### Scenario: 轻量融合
- **WHEN** 主干在任一 stage 进行条件融合
- **THEN** 默认采用轻量 gated residual fusion 或等价的高分辨率友好机制
- **AND** 明确融合发生在 attention 前后的位置
- **AND** 若启用 cross-attention，其计算开销需受控且仅作为可选增强项

### Requirement: 全局时间步嵌入
系统 SHALL 引入 time embedding，并在每个 encoder、latent 和 decoder stage 中显式使用，以支持一致性蒸馏中的不同时间步输入。

#### Scenario: 时间步调制
- **WHEN** 输入一致性时间步 `t`
- **THEN** 模型生成稳定的 time embedding
- **AND** 该 embedding 通过 bias、scale-shift、门控或等效方式注入每个 stage
- **AND** 不同时间步下模型输出空间保持一致且数值稳定

### Requirement: 长程依赖与高分辨率恢复
系统 SHALL 在轻量化前提下保留对长程依赖的建模能力，并支持高分辨率图像恢复。

#### Scenario: 高分辨率输入
- **WHEN** 输入尺寸从 256x256 扩展到更大分辨率
- **THEN** 网络模块应避免明显随分辨率平方爆炸的设计
- **AND** attention 与条件注入机制需优先采用适合恢复任务的高分辨率友好实现
- **AND** skip connection、归一化和上采样策略需保持稳定

### Requirement: 稳定的 x_0 预测输出
系统 SHALL 默认采用 `x_0` 预测作为主输出，并可选提供 residual 或 color correction 辅助头。

#### Scenario: 主输出设计
- **WHEN** 执行前向推理
- **THEN** 主输出为增强图像 `x_0_pred`
- **AND** 可选输出颜色校正参数图、残差图或置信度相关辅助结果
- **AND** 辅助头不得明显破坏主干轻量化目标

### Requirement: 可训练的完整损失体系
系统 SHALL 提供适用于水下图像增强一致性蒸馏的完整损失方案，并说明每一项的作用、推荐权重与适用条件。

#### Scenario: 基础损失组合
- **WHEN** 训练学生模型
- **THEN** 至少包含 `consistency loss`、重建损失、结构损失、感知损失和颜色校正损失
- **AND** 每项损失需给出推荐权重范围与启用建议

#### Scenario: 可选先验约束
- **WHEN** 数据集存在明显退化模式、颜色偏移或纹理损伤
- **THEN** 可选启用频域约束、边缘保持损失、measurement consistency、degradation consistency 或其他水下先验约束
- **AND** 需说明这些损失何时值得启用以及对训练稳定性的影响

### Requirement: 可直接落地的工程实现说明
系统 SHALL 产出可直接作为工程起点的 PyTorch 实现方案说明。

#### Scenario: 工程可实现性
- **WHEN** 交付最终实现
- **THEN** 输出需包含完整模块划分、forward 数据流、每个 stage 的输入输出维度
- **AND** 明确 attention、FFN、skip connection、condition fusion 的具体位置
- **AND** 代码可直接用于训练与测试的工程起点

## MODIFIED Requirements
### Requirement: 学生模型复杂度预算
学生模型的复杂度约束由“尽可能缩小参数量”修改为“在少步一致性蒸馏质量与轻量化之间平衡”，优先保证 1~4 步推理下的恢复质量、时间步一致性和高分辨率可扩展性。

## REMOVED Requirements
### Requirement: 纯卷积式学生主干优先
**Reason**: 纯卷积结构虽然更轻，但会明显削弱长程依赖建模，不利于水下图像增强中的全局颜色校正与大范围雾化补偿。
**Migration**: 改为采用 Restormer-lite 风格 transformer 主干，并通过减少 blocks、缩小通道和轻量融合模块控制复杂度。
