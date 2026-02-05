"""Prior specification helpers for Stan templates.

The Stan templates shipped with this project accept priors as *data* so that
YAML/JSON configuration can fully determine the prior families and parameters
without regenerating Stan code.

Notes
-----
Each prior is represented by:

* a family code (integer)
* up to 3 floating-point parameters (``p1``, ``p2``, ``p3``)

The mapping from family names to integer codes is defined by ``FAMILY_CODE`` and
must stay in sync with ``estimators/stan/common/prior_functions.stan``.

Examples
--------
Parse a YAML-like prior mapping:

>>> from comp_model_impl.estimators.stan.priors import parse_prior
>>> parse_prior({"family": "normal", "mu": 0.0, "sigma": 1.0})
Prior(family='normal', p1=0.0, p2=1.0, p3=0.0)

Convert a priors config into Stan data:

>>> from comp_model_impl.estimators.stan.priors import priors_to_stan_data
>>> class DummyAdapter:
...     def required_priors(self, family): return ["alpha"]
>>> priors_to_stan_data(priors_cfg={"alpha": {"family": "beta", "a": 2, "b": 2}},
...                     adapter=DummyAdapter(), family="indiv")
{'alpha_prior_family': 1, 'alpha_prior_p1': 2.0, 'alpha_prior_p2': 2.0, 'alpha_prior_p3': 0.0}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .adapters.base import StanAdapter


FAMILY_CODE = {
    "beta": 1,
    "normal": 2,
    "lognormal": 3,
    "gamma": 4,
    "exponential": 5,
    "half-normal": 6,
    "half_normal": 6,
    "student-t": 7,
    "student_t": 7,
    "t": 7,
    "cauchy": 8,
}


@dataclass(frozen=True)
class Prior:
    """Structured prior specification.

    Parameters
    ----------
    family : str
        Distribution family name. Supported: ``beta``, ``normal``,
        ``lognormal``, ``gamma``, ``exponential``, ``half-normal``,
        ``student-t``, ``cauchy``.
    p1 : float
        First family parameter in the order expected by Stan.
    p2 : float, optional
        Second family parameter (default: ``0.0``).
    p3 : float, optional
        Third family parameter (default: ``0.0``).
    """

    family: str
    p1: float
    p2: float = 0.0
    p3: float = 0.0


def _get(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first matching key in a mapping, or a default.

    Parameters
    ----------
    d : Mapping[str, Any]
        Source mapping.
    *keys : str
        Candidate keys to try in order.
    default : Any, optional
        Default value if none of the keys are present.

    Returns
    -------
    Any
        Value from the first matching key, or ``default`` if none are present.
    """
    for k in keys:
        if k in d:
            return d[k]
    return default


def parse_prior(d: Mapping[str, Any]) -> Prior:
    """Parse a config mapping into a :class:`Prior`.

    Parameters
    ----------
    d : Mapping[str, Any]
        Prior configuration. Must include a ``family`` field.

    Returns
    -------
    Prior
        Parsed prior.

    Raises
    ------
    ValueError
        If the family is unsupported or required parameters are missing.
    """
    fam = str(d["family"]).lower()
    if fam not in FAMILY_CODE:
        raise ValueError(f"Unsupported prior family {fam!r}. Supported: {sorted(set(FAMILY_CODE))}")

    if fam == "beta":
        a = float(_get(d, "a", "alpha"))
        b = float(_get(d, "b", "beta"))
        return Prior(fam, a, b, 0.0)

    if fam == "normal":
        return Prior(fam, float(_get(d, "mu", default=0.0)), float(_get(d, "sigma", default=1.0)), 0.0)

    if fam == "lognormal":
        return Prior(fam, float(_get(d, "mu", default=0.0)), float(_get(d, "sigma", default=1.0)), 0.0)

    if fam == "gamma":
        return Prior(fam, float(_get(d, "shape")), float(_get(d, "rate")), 0.0)

    if fam == "exponential":
        return Prior(fam, float(_get(d, "rate", default=1.0)), 0.0, 0.0)

    if fam in {"half-normal", "half_normal"}:
        return Prior("half-normal", float(_get(d, "sigma", default=1.0)), 0.0, 0.0)

    if fam in {"student-t", "student_t", "t"}:
        df = float(_get(d, "df"))
        mu = float(_get(d, "mu", default=0.0))
        sigma = float(_get(d, "sigma", default=1.0))
        return Prior("student-t", df, mu, sigma)

    if fam == "cauchy":
        loc = float(_get(d, "loc", "mu", default=0.0))
        scale = float(_get(d, "scale", "sigma", default=1.0))
        return Prior(fam, loc, scale, 0.0)

    raise AssertionError("unreachable")


def priors_to_stan_data(
    *,
    priors_cfg: Mapping[str, Any] | None,
    adapter: StanAdapter,
    family: str,
    forbid_extra: bool = False,
) -> dict[str, Any]:
    """Convert a priors config into a Stan ``data`` dictionary.

    Parameters
    ----------
    priors_cfg : Mapping[str, Any] or None
        Mapping from prior name to config dict or :class:`Prior`.
    adapter : comp_model_impl.estimators.stan.adapters.base.StanAdapter
        Adapter defining required priors for the selected Stan family.
    family : str
        Stan family identifier (e.g., ``"indiv"``, ``"hier"``).
    forbid_extra : bool, optional
        If ``True``, raise when priors not required by the adapter are provided.

    Returns
    -------
    dict[str, Any]
        Flat mapping of Stan data fields for priors (``*_prior_family``,
        ``*_prior_p1``, ``*_prior_p2``, ``*_prior_p3``).

    See Also
    --------
    priors_to_stan_data_strict
    """
    required: Sequence[str] = adapter.required_priors(family)
    return priors_to_stan_data_strict(priors_cfg=priors_cfg, required=required, forbid_extra=forbid_extra)


def priors_to_stan_data_strict(
    *,
    priors_cfg: Mapping[str, Any] | None,
    required: Sequence[str],
    forbid_extra: bool = False,
) -> dict[str, Any]:
    """Convert priors config into Stan ``data`` given an explicit required list.

    Parameters
    ----------
    priors_cfg : Mapping[str, Any] or None
        Mapping from prior name to config dict or :class:`Prior`.
    required : Sequence[str]
        Explicit list of required prior names.
    forbid_extra : bool, optional
        If ``True``, raise when priors not listed in ``required`` are provided.

    Returns
    -------
    dict[str, Any]
        Flat mapping of Stan data fields for priors.

    Raises
    ------
    ValueError
        If required priors are missing or extras are provided when forbidden.

    See Also
    --------
    priors_to_stan_data
    """
    if priors_cfg is None:
        raise ValueError(f"Missing priors config. Required: {list(required)}")

    missing = [k for k in required if k not in priors_cfg]
    if missing:
        raise ValueError(f"Missing required priors: {missing}")

    if forbid_extra:
        extra = [k for k in priors_cfg.keys() if k not in required]
        if extra:
            raise ValueError(f"Unknown priors provided (not used): {extra}")

    out: dict[str, Any] = {}
    for name in required:
        v = priors_cfg[name]
        pr = v if isinstance(v, Prior) else parse_prior(v)

        fam = pr.family.lower()
        if fam not in FAMILY_CODE:
            raise ValueError(f"Unsupported prior family {fam!r} for {name!r}")

        out[f"{name}_prior_family"] = int(FAMILY_CODE[fam])
        out[f"{name}_prior_p1"] = float(pr.p1)
        out[f"{name}_prior_p2"] = float(pr.p2)
        out[f"{name}_prior_p3"] = float(pr.p3)

    return out
