from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

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
    family: str
    p1: float
    p2: float = 0.0
    p3: float = 0.0

def _get(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default

def parse_prior(d: Mapping[str, Any]) -> Prior:
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
        # Stan uses shape, rate
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

def priors_to_stan_data(priors_cfg: Mapping[str, Any] | None, defaults: Mapping[str, Prior]) -> dict[str, Any]:
    priors_cfg = priors_cfg or {}
    out: dict[str, Any] = {}
    for name, default in defaults.items():
        pr = default
        if name in priors_cfg:
            pr = parse_prior(priors_cfg[name])

        out[f"{name}_prior_family"] = int(FAMILY_CODE[pr.family])
        out[f"{name}_prior_p1"] = float(pr.p1)
        out[f"{name}_prior_p2"] = float(pr.p2)
        out[f"{name}_prior_p3"] = float(pr.p3)
    return out
