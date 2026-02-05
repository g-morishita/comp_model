"""Estimators implemented in :mod:`comp_model_impl`.

The core package (:mod:`comp_model_core`) only defines the estimator interface.
This subpackage provides concrete implementations such as MLE and Stan-based
Bayesian estimators.

Notes
-----
Stan estimators and their templates live in :mod:`comp_model_impl.estimators.stan`.
Importing from this module re-exports the primary Stan estimator classes for
convenience.

Examples
--------
>>> from comp_model_impl.estimators import BoxMLESubjectwiseEstimator
>>> from comp_model_impl.estimators import StanNUTSSubjectwiseEstimator
"""

from .mle_event_log import BoxMLESubjectwiseEstimator, TransformedMLESubjectwiseEstimator
from .stan import StanHierarchicalNUTSEstimator, StanNUTSSubjectwiseEstimator
from .within_subject_shared_delta import WithinSubjectSharedDeltaTransformedMLEEstimator

__all__ = [
    "BoxMLESubjectwiseEstimator",
    "TransformedMLESubjectwiseEstimator",
    "StanHierarchicalNUTSEstimator",
    "StanNUTSSubjectwiseEstimator",
    "WithinSubjectSharedDeltaTransformedMLEEstimator",
]
