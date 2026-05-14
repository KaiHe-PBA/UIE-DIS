# Tasks
- [x] Task 1: 明确工程目录与脚本边界
  - [x] SubTask 1.1: 定义 `models`、`losses`、`datasets`、`utils`、`train.py`、`eval.py`、`infer.py` 的文件职责
  - [x] SubTask 1.2: 约定配置项，包括输入尺寸、通道基数、时间步嵌入维度、损失权重与训练超参数
  - [x] SubTask 1.3: 约定模型前向接口，统一 `x_t`、`y`、`t` 与可选返回项格式

- [x] Task 2: 设计轻量学生模型主干
  - [x] SubTask 2.1: 选定主干为 NAFNet-style U-Net 或更适合水下增强的轻量变体，并给出选择理由
  - [x] SubTask 2.2: 设计显式 `time embedding` 注入路径，并明确各尺度如何使用
  - [x] SubTask 2.3: 设计独立 `condition encoder`，输出多尺度条件特征
  - [x] SubTask 2.4: 设计每个尺度的条件融合模块，优先选择轻量稳定的融合方式
  - [x] SubTask 2.5: 设计主输出、残差辅助输出与可选轻量颜色校正分支
  - [x] SubTask 2.6: 检查参数量与计算量预算，保证默认配置下优先控制在 3M 以内

- [x] Task 3: 设计面向水下增强的结构改造
  - [x] SubTask 3.1: 识别水下图像常见退化，如偏色、低对比度、雾化散射与细节丢失
  - [x] SubTask 3.2: 选择轻量改造模块，如颜色引导、边缘增强、频率补偿或 gated 调制
  - [x] SubTask 3.3: 将改造体现在网络结构与前向流程说明中，并说明其必要性

- [x] Task 4: 编写可工程化的 PyTorch 模型代码骨架
  - [x] SubTask 4.1: 输出模块级类定义，包括基础块、时间嵌入、条件编码器、融合块与 U-Net 主体
  - [x] SubTask 4.2: 标注主要张量维度流、通道变化与上采样/下采样逻辑
  - [x] SubTask 4.3: 提供规范命名、必要注释与清晰的 `forward` 逻辑
  - [x] SubTask 4.4: 给出默认超参数建议，如 `base_channels=32` 或 `48` 的取舍依据

- [x] Task 5: 设计完整训练损失
  - [x] SubTask 5.1: 定义 consistency loss，并说明与 teacher 或相邻时间步目标的关系
  - [x] SubTask 5.2: 定义重建损失，如 L1 或 Charbonnier
  - [x] SubTask 5.3: 定义结构约束，如 SSIM
  - [x] SubTask 5.4: 定义可选感知损失、颜色一致性损失、边缘损失与频域损失
  - [x] SubTask 5.5: 说明每类损失的启用条件、作用与推荐权重范围

- [x] Task 6: 设计数据集与训练验证流程
  - [x] SubTask 6.1: 设计训练数据集接口，支持成对数据、teacher 监督信息与时间步采样
  - [x] SubTask 6.2: 设计训练循环，包括前向、损失汇总、反向传播、日志与 checkpoint 保存
  - [x] SubTask 6.3: 设计验证循环，包括 PSNR、SSIM、UIQM 或 UCIQE 等指标接口
  - [x] SubTask 6.4: 设计推理脚本，覆盖一步与少步一致性采样

- [x] Task 7: 输出最终交付文档内容
  - [x] SubTask 7.1: 按“整体架构概述 -> 模块级设计 -> PyTorch 代码骨架 -> 完整训练损失定义 -> 推理流程”组织输出
  - [x] SubTask 7.2: 确保内容足够具体，可直接改写为工程代码
  - [x] SubTask 7.3: 复核术语、一致性蒸馏表述与脚本命名清晰度

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 2
- Task 5 depends on Task 2
- Task 6 depends on Task 1, Task 4, Task 5
- Task 7 depends on Task 3, Task 4, Task 5, Task 6
