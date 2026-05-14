from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

from .losses import LossWeights


ResourceLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class LossApplicability:
    requires_target_x0: bool = False
    requires_teacher_x0_or_output: bool = False
    requires_measurement: bool = False
    requires_measurement_operator: bool = False
    requires_degraded_target_or_y: bool = False
    requires_degradation_operator: bool = False
    optional_output_keys: tuple[str, ...] = ()
    recommended_when: tuple[str, ...] = ()
    avoid_when: tuple[str, ...] = ()


@dataclass(frozen=True)
class NumericalStabilityAdvice:
    recommended_input_range: Optional[tuple[float, float]] = (0.0, 1.0)
    amp_safe: bool = True
    prefer_float32: bool = False
    default_epsilon: Optional[float] = None
    tips: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceTradeoff:
    memory_cost: ResourceLevel
    speed_cost: ResourceLevel
    effect_gain: ResourceLevel
    summary: str
    lightweight_alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class LossRecommendation:
    name: str
    weight_field: str
    description: str
    applicability: LossApplicability
    stability: NumericalStabilityAdvice
    tradeoff: ResourceTradeoff
    suggested_weight_range: tuple[float, float]


@dataclass(frozen=True)
class RecommendedLossSetup:
    profile: str
    weights: LossWeights
    perceptual_backend: str
    enabled_losses: tuple[str, ...]
    disabled_losses: tuple[str, ...]
    rationale: tuple[str, ...]
    further_lightweight_suggestions: tuple[str, ...]


def _build_loss_catalog() -> Dict[str, LossRecommendation]:
    return {
        "consistency": LossRecommendation(
            name="consistency",
            weight_field="consistency",
            description="利用 teacher 或额外参考分支约束当前 x0 预测，适合蒸馏、多步一致性和 teacher-student 训练。",
            applicability=LossApplicability(
                requires_teacher_x0_or_output=True,
                optional_output_keys=("residual_pred",),
                recommended_when=("使用 teacher/student 训练", "需要约束多步预测一致性"),
                avoid_when=("没有 teacher/reference 信号",),
            ),
            stability=NumericalStabilityAdvice(
                default_epsilon=1e-3,
                tips=("teacher 输出尽量 stop-gradient", "mixed precision 下建议保留 epsilon>=1e-3"),
            ),
            tradeoff=ResourceTradeoff(
                memory_cost="low",
                speed_cost="low",
                effect_gain="high",
                summary="额外开销小，通常是最先保留的稳定项。",
                lightweight_alternatives=("仅保留主 consistency，关闭 residual_cycle_weight",),
            ),
            suggested_weight_range=(0.5, 2.0),
        ),
        "charbonnier": LossRecommendation(
            name="charbonnier",
            weight_field="charbonnier",
            description="对像素重建更稳健的基础项，通常作为默认主重建损失。",
            applicability=LossApplicability(
                requires_target_x0=True,
                recommended_when=("存在 GT x0", "希望比 L1 对异常值更稳健"),
                avoid_when=("没有监督 target_x0",),
            ),
            stability=NumericalStabilityAdvice(
                default_epsilon=1e-3,
                tips=("输入先归一化到 [0, 1] 或 [-1, 1]", "半精度下 epsilon 可增大到 1e-2"),
            ),
            tradeoff=ResourceTradeoff(
                memory_cost="low",
                speed_cost="low",
                effect_gain="high",
                summary="重建质量稳定，几乎没有额外资源负担。",
                lightweight_alternatives=("可单独保留此项作为最小训练配置",),
            ),
            suggested_weight_range=(0.5, 2.0),
        ),
        "l1": LossRecommendation(
            name="l1",
            weight_field="l1",
            description="像素级绝对误差，适合做辅助项或与 Charbonnier 二选一。",
            applicability=LossApplicability(
                requires_target_x0=True,
                recommended_when=("希望更直接的像素约束",),
                avoid_when=("噪声/离群点较多且训练易抖动",),
            ),
            stability=NumericalStabilityAdvice(
                tips=("与 Charbonnier 同开时通常降低其权重",),
            ),
            tradeoff=ResourceTradeoff(
                memory_cost="low",
                speed_cost="low",
                effect_gain="medium",
                summary="简单直接，但通常不如 Charbonnier 稳健。",
                lightweight_alternatives=("优先保留 Charbonnier，必要时关闭 L1",),
            ),
            suggested_weight_range=(0.0, 0.5),
        ),
        "ssim": LossRecommendation(
            name="ssim",
            weight_field="ssim",
            description="强调局部结构与亮度对比一致性，适合纹理和结构恢复。",
            applicability=LossApplicability(
                requires_target_x0=True,
                recommended_when=("关心结构相似度和局部对比度",),
                avoid_when=("patch 很小或输入动态范围未对齐",),
            ),
            stability=NumericalStabilityAdvice(
                recommended_input_range=(0.0, 1.0),
                tips=("data_range 必须与输入范围匹配", "小分辨率 patch 可把 window_size 从 11 降到 7"),
            ),
            tradeoff=ResourceTradeoff(
                memory_cost="medium",
                speed_cost="medium",
                effect_gain="medium",
                summary="结构恢复有帮助，但卷积窗口带来额外计算。",
                lightweight_alternatives=("低资源时只保留小权重 SSIM", "仍不够快时优先关闭 SSIM"),
            ),
            suggested_weight_range=(0.1, 0.5),
        ),
        "perceptual": LossRecommendation(
            name="perceptual",
            weight_field="perceptual",
            description="利用多层视觉特征增强感知质量，适合追求主观视觉效果。",
            applicability=LossApplicability(
                requires_target_x0=True,
                recommended_when=("更重视视觉观感而非纯像素指标", "有 3 通道 RGB 输入"),
                avoid_when=("显存很紧张", "训练速度优先"),
            ),
            stability=NumericalStabilityAdvice(
                recommended_input_range=(0.0, 1.0),
                tips=("VGG 后端要求 RGB 3 通道", "AMP 下若特征抖动可切回 lightweight 后端"),
            ),
            tradeoff=ResourceTradeoff(
                memory_cost="high",
                speed_cost="high",
                effect_gain="high",
                summary="效果提升明显，但 VGG 感知特征是最重的项之一。",
                lightweight_alternatives=("改用 lightweight backend", "降低 perceptual 权重", "关闭该项"),
            ),
            suggested_weight_range=(0.02, 0.2),
        ),
        "color": LossRecommendation(
            name="color",
            weight_field="color",
            description="约束通道均值/方差并可正则 color_map，适合水下色偏校正。",
            applicability=LossApplicability(
                requires_target_x0=True,
                optional_output_keys=("color_map",),
                recommended_when=("存在明显颜色偏移", "模型输出 color_map 需要平滑约束"),
                avoid_when=("颜色分布天然不稳定且无 GT 颜色参考",),
            ),
            stability=NumericalStabilityAdvice(
                tips=("输入范围尽量固定", "若 color_map 波动过大可先减小 color_map_weight"),
            ),
            tradeoff=ResourceTradeoff(
                memory_cost="low",
                speed_cost="low",
                effect_gain="medium",
                summary="额外成本低，水下增强场景通常值得保留。",
                lightweight_alternatives=("只保留均值/方差约束，不使用 color_map 正则",),
            ),
            suggested_weight_range=(0.05, 0.3),
        ),
        "frequency": LossRecommendation(
            name="frequency",
            weight_field="frequency",
            description="在频域约束幅值分布，适合细节与纹理恢复，但 FFT 开销较高。",
            applicability=LossApplicability(
                requires_target_x0=True,
                recommended_when=("需要增强高频细节", "纹理恢复不足"),
                avoid_when=("显存或吞吐优先", "超小 batch 且 AMP 不稳定"),
            ),
            stability=NumericalStabilityAdvice(
                amp_safe=False,
                prefer_float32=True,
                tips=("FFT 建议在 float32 中计算", "保留 log1p 幅值可抑制大值主导"),
            ),
            tradeoff=ResourceTradeoff(
                memory_cost="high",
                speed_cost="high",
                effect_gain="medium",
                summary="对细节有帮助，但通常是最先关闭的重项之一。",
                lightweight_alternatives=("关闭 frequency", "以 edge 代替部分高频约束"),
            ),
            suggested_weight_range=(0.01, 0.1),
        ),
        "edge": LossRecommendation(
            name="edge",
            weight_field="edge",
            description="利用 Sobel 边缘约束几何轮廓，适合边界模糊的恢复任务。",
            applicability=LossApplicability(
                requires_target_x0=True,
                recommended_when=("边缘发虚", "希望强化轮廓"),
                avoid_when=("噪声很重且边缘本身不可靠",),
            ),
            stability=NumericalStabilityAdvice(
                tips=("高噪声数据建议搭配 Charbonnier 并降低 edge 权重",),
            ),
            tradeoff=ResourceTradeoff(
                memory_cost="low",
                speed_cost="medium",
                effect_gain="medium",
                summary="比频域更轻，常作为细节增强的折中项。",
                lightweight_alternatives=("只保留 edge，不启用 frequency",),
            ),
            suggested_weight_range=(0.02, 0.15),
        ),
        "measurement": LossRecommendation(
            name="measurement",
            weight_field="measurement",
            description="要求预测结果经过观测算子后匹配测量值，适合逆问题和物理一致性约束。",
            applicability=LossApplicability(
                requires_measurement=True,
                requires_measurement_operator=True,
                recommended_when=("存在可微观测模型与测量值",),
                avoid_when=("没有可微 operator", "测量尺度未校准"),
            ),
            stability=NumericalStabilityAdvice(
                tips=("operator 输出尺度需与 measurement 对齐", "不稳定时先只保留低权重 measurement"),
            ),
            tradeoff=ResourceTradeoff(
                memory_cost="medium",
                speed_cost="medium",
                effect_gain="high",
                summary="物理一致性强，但算子本身可能带来非小开销。",
                lightweight_alternatives=("预先缓存 operator 的固定部分",),
            ),
            suggested_weight_range=(0.05, 0.5),
        ),
        "degradation": LossRecommendation(
            name="degradation",
            weight_field="degradation",
            description="要求预测退化后与目标退化域匹配，适合已知退化模型的恢复任务。",
            applicability=LossApplicability(
                requires_degraded_target_or_y=True,
                requires_degradation_operator=True,
                recommended_when=("已知退化过程并希望约束可逆性",),
                avoid_when=("退化模型误差较大",),
            ),
            stability=NumericalStabilityAdvice(
                tips=("先校验退化算子是否保持 batch/channel 维度一致",),
            ),
            tradeoff=ResourceTradeoff(
                memory_cost="medium",
                speed_cost="medium",
                effect_gain="high",
                summary="先验充足时收益明显，否则可能引入错误偏置。",
                lightweight_alternatives=("直接复用 y 作为 degraded_target，避免重复生成",),
            ),
            suggested_weight_range=(0.05, 0.5),
        ),
    }


_LOSS_CATALOG = _build_loss_catalog()
_LOSS_FIELDS = tuple(_LOSS_CATALOG.keys())


def get_loss_recommendation_catalog() -> Dict[str, LossRecommendation]:
    return dict(_LOSS_CATALOG)


def get_loss_recommendation(loss_name: str) -> LossRecommendation:
    try:
        return _LOSS_CATALOG[loss_name]
    except KeyError as exc:
        raise KeyError(f"Unknown loss recommendation: {loss_name}") from exc


def get_active_loss_names(weights: LossWeights) -> tuple[str, ...]:
    return tuple(name for name in _LOSS_FIELDS if getattr(weights, name) > 0)


def validate_active_loss_requirements(
    weights: LossWeights,
    *,
    target_x0_available: bool = False,
    teacher_x0_available: bool = False,
    teacher_output_available: bool = False,
    measurement_available: bool = False,
    degraded_target_available: bool = False,
    y_available: bool = False,
    measurement_operator_available: bool = False,
    degradation_operator_available: bool = False,
) -> Dict[str, tuple[str, ...]]:
    issues: Dict[str, tuple[str, ...]] = {}
    for loss_name in get_active_loss_names(weights):
        applicability = _LOSS_CATALOG[loss_name].applicability
        current_issues = []
        if applicability.requires_target_x0 and not target_x0_available:
            current_issues.append("缺少 `target_x0`，当前损失无法生效。")
        if applicability.requires_teacher_x0_or_output and not (
            teacher_x0_available or teacher_output_available
        ):
            current_issues.append("缺少 `teacher_x0` 或 `teacher_output`。")
        if applicability.requires_measurement and not measurement_available:
            current_issues.append("缺少 `measurement`。")
        if applicability.requires_measurement_operator and not measurement_operator_available:
            current_issues.append("缺少 `measurement_operator`。")
        if applicability.requires_degraded_target_or_y and not (degraded_target_available or y_available):
            current_issues.append("缺少 `degraded_target` 或可复用的 `y`。")
        if applicability.requires_degradation_operator and not degradation_operator_available:
            current_issues.append("缺少 `degradation_operator`。")
        if current_issues:
            issues[loss_name] = tuple(current_issues)
    return issues


def build_recommended_loss_setup(
    profile: str = "balanced",
    *,
    include_measurement: bool = False,
    include_degradation: bool = False,
    prefer_lightweight_perceptual: Optional[bool] = None,
) -> RecommendedLossSetup:
    normalized_profile = profile.lower()
    if normalized_profile == "balanced":
        weights = LossWeights(consistency=1.0, charbonnier=1.0, ssim=0.35, perceptual=0.08, color=0.2)
        perceptual_backend = "auto"
        rationale = (
            "兼顾像素、结构和感知质量，适合作为默认起点。",
            "保留轻量可退化的 perceptual 项，避免一开始就把频域项堆满。",
        )
        lightweight = (
            "显存紧张时先把 perceptual backend 固定为 lightweight。",
            "再关闭 SSIM 或把其权重降到 0.1。",
            "frequency 与 edge 暂不启用，避免无谓增加训练成本。",
        )
    elif normalized_profile == "stability_first":
        weights = LossWeights(consistency=1.0, charbonnier=1.0, ssim=0.2, perceptual=0.0, color=0.15)
        perceptual_backend = "lightweight"
        rationale = (
            "优先使用最稳的重建与颜色项，避免高波动的重损失。",
            "适合数据尺度还在调试、AMP 刚接入或 batch 很小的阶段。",
        )
        lightweight = (
            "保留 Charbonnier 作为主项，避免同时堆高 L1 与 SSIM。",
            "保持 FFT 相关损失关闭，待训练稳定后再逐步加入。",
            "必要时把颜色项降到 0.05 先保证收敛。",
        )
    elif normalized_profile == "memory_efficient":
        weights = LossWeights(consistency=1.0, charbonnier=1.0, ssim=0.15, perceptual=0.0, color=0.1)
        perceptual_backend = "lightweight"
        rationale = (
            "最小化显存和前向开销，优先保留收益/成本比最高的基础项。",
            "适合低显存卡、较大分辨率或需要提高吞吐的训练。",
        )
        lightweight = (
            "保持 perceptual/frequency/edge 关闭。",
            "优先使用裁剪训练或更小 patch，再考虑降低 batch。",
            "如仍超显存，可把 SSIM 也关闭，仅保留 consistency+charbonnier+color。",
        )
    elif normalized_profile == "quality_first":
        weights = LossWeights(
            consistency=1.0,
            charbonnier=1.0,
            ssim=0.5,
            perceptual=0.12,
            color=0.2,
            frequency=0.05,
            edge=0.05,
        )
        perceptual_backend = "torchvision_vgg16"
        rationale = (
            "优先追求主观视觉效果和细节恢复。",
            "在充足显存下同时引入感知、结构和细节增强项。",
        )
        lightweight = (
            "若 VGG 后端过重，先切换为 lightweight perceptual 而不是直接关掉感知项。",
            "若吞吐下降明显，优先关闭 frequency，再考虑降低 edge。",
            "FFT 在 AMP 不稳时建议改为 float32 或直接关闭。",
        )
    elif normalized_profile == "fast_train":
        weights = LossWeights(consistency=1.0, charbonnier=1.0, ssim=0.1, perceptual=0.0, color=0.1)
        perceptual_backend = "lightweight"
        rationale = (
            "优先提高 step/s，适合快速试参和回归验证。",
            "只保留最便宜且最常有效的重建项。",
        )
        lightweight = (
            "只在训练后期按需补开 SSIM 或 perceptual。",
            "吞吐仍不足时把颜色项权重减半或关闭。",
            "保留 consistency 可减少快速试参时的发散风险。",
        )
    else:
        raise ValueError(
            "Unknown profile. Expected one of: balanced, stability_first, memory_efficient, quality_first, fast_train."
        )

    if include_measurement:
        weights.measurement = 0.2
    if include_degradation:
        weights.degradation = 0.2
    if prefer_lightweight_perceptual is True:
        perceptual_backend = "lightweight"
    enabled_losses = get_active_loss_names(weights)
    disabled_losses = tuple(name for name in _LOSS_FIELDS if name not in enabled_losses)
    return RecommendedLossSetup(
        profile=normalized_profile,
        weights=weights,
        perceptual_backend=perceptual_backend,
        enabled_losses=enabled_losses,
        disabled_losses=disabled_losses,
        rationale=rationale,
        further_lightweight_suggestions=lightweight,
    )
