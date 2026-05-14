import torch

from uie_consistency import (
    LossWeights,
    UnderwaterLossAssembler,
    build_recommended_loss_setup,
    get_active_loss_names,
    get_loss_recommendation,
    get_loss_recommendation_catalog,
    validate_active_loss_requirements,
)


def test_loss_recommendation_catalog_exposes_conditions_and_tradeoffs():
    catalog = get_loss_recommendation_catalog()

    assert "frequency" in catalog
    assert "perceptual" in catalog
    assert catalog["frequency"].stability.prefer_float32 is True
    assert catalog["perceptual"].tradeoff.memory_cost == "high"
    assert catalog["consistency"].applicability.requires_teacher_x0_or_output is True


def test_build_recommended_loss_setup_memory_efficient_is_directly_usable():
    setup = build_recommended_loss_setup("memory_efficient")

    assert setup.perceptual_backend == "lightweight"
    assert "frequency" not in setup.enabled_losses
    assert "perceptual" not in setup.enabled_losses
    assert get_active_loss_names(setup.weights) == setup.enabled_losses

    assembler = UnderwaterLossAssembler(weights=setup.weights, perceptual_backend=setup.perceptual_backend)
    output = {"x0_pred": torch.full((2, 3, 8, 8), 0.6)}
    target = torch.full((2, 3, 8, 8), 0.4)
    teacher = torch.full((2, 3, 8, 8), 0.5)

    terms = assembler(output, target_x0=target, teacher_x0=teacher)

    assert set(("total", "consistency", "charbonnier", "ssim", "color")).issubset(terms.keys())
    assert "perceptual" not in terms
    assert terms["total"].ndim == 0


def test_validate_active_loss_requirements_reports_missing_inputs():
    weights = LossWeights(consistency=1.0, charbonnier=1.0, measurement=0.2, degradation=0.2)

    issues = validate_active_loss_requirements(weights)

    assert issues["consistency"] == ("缺少 `teacher_x0` 或 `teacher_output`。",)
    assert issues["charbonnier"] == ("缺少 `target_x0`，当前损失无法生效。",)
    assert "缺少 `measurement`。" in issues["measurement"]
    assert "缺少 `measurement_operator`。" in issues["measurement"]
    assert "缺少 `degraded_target` 或可复用的 `y`。" in issues["degradation"]
    assert "缺少 `degradation_operator`。" in issues["degradation"]


def test_validate_active_loss_requirements_accepts_available_runtime_inputs():
    setup = build_recommended_loss_setup("balanced", include_measurement=True, include_degradation=True)
    recommendation = get_loss_recommendation("measurement")

    issues = validate_active_loss_requirements(
        setup.weights,
        target_x0_available=True,
        teacher_output_available=True,
        measurement_available=True,
        degraded_target_available=True,
        measurement_operator_available=True,
        degradation_operator_available=True,
    )

    assert issues == {}
    assert recommendation.tradeoff.effect_gain == "high"
