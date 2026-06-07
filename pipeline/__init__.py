"""SCFE measurement pipeline package."""

__all__ = [
    "PipelineError",
    "PipelineResult",
    "ModelRegistry",
    "run_scfe_pipeline",
]


def __getattr__(name: str):
    """Lazy exports so lightweight imports (e.g. config) avoid loading torch."""
    if name == "run_scfe_pipeline":
        from pipeline.inference import run_scfe_pipeline

        return run_scfe_pipeline
    if name == "PipelineError":
        from pipeline.inference import PipelineError

        return PipelineError
    if name == "PipelineResult":
        from pipeline.inference import PipelineResult

        return PipelineResult
    if name == "ModelRegistry":
        from pipeline.model_registry import ModelRegistry

        return ModelRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
