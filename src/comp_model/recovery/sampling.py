from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .priors import NormalPrior, Transform


@dataclass(frozen=True, slots=True)
class IndependentSubjectSampler:
    """
    Sample subject params directly (no population layer).
    Example: alpha ~ logistic(N(0,1)), beta ~ softplus(N(1,0.5))
    """
    latent_priors: Mapping[str, NormalPrior]      # per param latent prior
    transforms: Mapping[str, Transform]           # per param transform

    def sample_subject_params(self, rng: np.random.Generator, subject_ids: Sequence[str]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for sid in subject_ids:
            out[sid] = {}
            for p, prior in self.latent_priors.items():
                z = prior.sample(rng)
                out[sid][p] = float(self.transforms[p].forward(z))
        return out


@dataclass(frozen=True, slots=True)
class HierarchicalSampler:
    """
    Sample population then individuals.
      mu_p ~ Normal(mu0, sd0)
      log_sigma_p ~ Normal(log(s0), sd1)  -> sigma_p = exp(log_sigma_p)
      z_{i,p} ~ Normal(mu_p, sigma_p)
      theta_{i,p} = transform_p(z_{i,p})
    """
    mu_priors: Mapping[str, NormalPrior]          # prior over mu_p (latent scale)
    log_sigma_priors: Mapping[str, NormalPrior]   # prior over log_sigma_p
    transforms: Mapping[str, Transform]

    def sample_population(self, rng: np.random.Generator) -> dict[str, float]:
        pop: dict[str, float] = {}
        for p, pr in self.mu_priors.items():
            pop[f"mu_{p}"] = pr.sample(rng)
        for p, pr in self.log_sigma_priors.items():
            pop[f"log_sigma_{p}"] = pr.sample(rng)
        return pop

    def sample_subject_params(
        self,
        rng: np.random.Generator,
        subject_ids: Sequence[str],
        population: Mapping[str, float],
    ) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for sid in subject_ids:
            out[sid] = {}
            for p in self.transforms.keys():
                mu = float(population[f"mu_{p}"])
                log_sigma = float(population[f"log_sigma_{p}"])
                sigma = float(np.exp(log_sigma))
                z = float(rng.normal(mu, sigma))
                out[sid][p] = float(self.transforms[p].forward(z))
        return out
