"""Chapter 6 overlap-aware dummy-clevis utilities."""

from .paths import REPO_ROOT, CHAPTER6_ROOT
from .sampler import RelaxedSample, sample_relaxed_params, validate_relaxed_params

__all__ = [
    "REPO_ROOT",
    "CHAPTER6_ROOT",
    "RelaxedSample",
    "sample_relaxed_params",
    "validate_relaxed_params",
]
