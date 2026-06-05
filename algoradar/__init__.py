"""AlgoRadar: ML-powered competitive programming analytics."""

__all__ = ["run_analysis"]


def __getattr__(name: str):
    if name == "run_analysis":
        from .pipeline import run_analysis

        return run_analysis
    raise AttributeError(f"module 'algoradar' has no attribute {name!r}")
