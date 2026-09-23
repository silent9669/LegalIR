from src.models.bootstrap import download_models, required_model_files

try:
    from src.models.device import resolve_device
    from src.models.parameter_audit import (
        DEFAULT_PIPELINE_MODELS,
        KNOWN_PARAM_COUNTS,
        MAX_PARAMETER_BUDGET,
        ParameterBudgetExceededError,
        audit_model_parameters,
        audit_system_parameters,
        count_parameters,
        count_parameters_from_config,
        estimate_transformer_parameters,
        extract_models_from_config,
        validate_parameter_budget,
    )
except ImportError:
    # Graceful degradation for lightweight CPU/bootstrap containers without torch
    resolve_device = None  # type: ignore[assignment]
    DEFAULT_PIPELINE_MODELS = []  # type: ignore[assignment]
    KNOWN_PARAM_COUNTS = {}  # type: ignore[assignment]
    MAX_PARAMETER_BUDGET = 4_000_000_000
    ParameterBudgetExceededError = RuntimeError  # type: ignore[assignment,misc]
    audit_model_parameters = None  # type: ignore[assignment]
    audit_system_parameters = None  # type: ignore[assignment]
    count_parameters = None  # type: ignore[assignment]
    count_parameters_from_config = None  # type: ignore[assignment]
    estimate_transformer_parameters = None  # type: ignore[assignment]
    extract_models_from_config = None  # type: ignore[assignment]
    validate_parameter_budget = None  # type: ignore[assignment]

__all__ = [
    "DEFAULT_PIPELINE_MODELS",
    "KNOWN_PARAM_COUNTS",
    "MAX_PARAMETER_BUDGET",
    "ParameterBudgetExceededError",
    "audit_model_parameters",
    "audit_system_parameters",
    "count_parameters",
    "count_parameters_from_config",
    "download_models",
    "estimate_transformer_parameters",
    "extract_models_from_config",
    "required_model_files",
    "resolve_device",
    "validate_parameter_budget",
]
