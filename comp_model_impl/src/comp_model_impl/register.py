"""comp_model_impl.register

Registry wiring for the implementation package.

This module registers concrete implementations (models, estimators, generators,
environments, and runtime tasks/runners) into a shared
:class:`comp_model_core.registry.Registry`.

See Also
--------
comp_model_core.registry.Registry
"""

from __future__ import annotations

from comp_model_core.registry import Registry

# models
from comp_model_impl.models.qrl.qrl import QRL
from comp_model_impl.models.vs.vs import VS
from comp_model_impl.models.vicarious_rl.vicarious_rl import Vicarious_RL

# estimators
from comp_model_impl.estimators.mle_event_log import BoxMLESubjectwiseEstimator, TransformedMLESubjectwiseEstimator
from comp_model_impl.estimators.stan.nuts import StanNUTSSubjectwiseEstimator

# generators
from comp_model_impl.generators.event_log import (
    EventLogAsocialGenerator,
    EventLogSocialPreChoiceGenerator,
    EventLogSocialPostOutcomeGenerator,
)
from comp_model_impl.generators.trial_by_trial import (
    AsocialBanditGenerator,
    SocialPreChoiceGenerator,
    SocialPostOutcomeGenerator,
)

# environments + block runners
from comp_model_impl.bandits.bernoulli import BernoulliBanditEnv
from comp_model_impl.tasks.runner_block_wrappers import BanditBlockRunner, SocialBanditBlockRunner


def register_all(reg: Registry) -> None:
    """Register all implementation classes into a registry.

    Parameters
    ----------
    reg : Registry
        Registry instance to populate.

    Notes
    -----
    This function registers:
    - models
    - estimators
    - generators
    - bandit environments
    - runtime tasks/runners

    The string keys used here are the public names that can be referenced in YAML/JSON
    plans or configuration.
    """
    # models
    reg.models.register("QRL", QRL)
    reg.models.register("VS", VS)
    reg.models.register("Vicarious_RL", Vicarious_RL)

    # estimators
    reg.estimators.register("BoxMLESubjectwiseEstimator", BoxMLESubjectwiseEstimator)
    reg.estimators.register("TransformedMLESubjectwiseEstimator", TransformedMLESubjectwiseEstimator)
    reg.estimators.register("StanNUTSSubjectwiseEstimator", StanNUTSSubjectwiseEstimator)

    # generators
    reg.generators.register("AsocialBanditGenerator", AsocialBanditGenerator)
    reg.generators.register("SocialPreChoiceGenerator", SocialPreChoiceGenerator)
    reg.generators.register("SocialPostOutcomeGenerator", SocialPostOutcomeGenerator)
    reg.generators.register("EventLogAsocialGenerator", EventLogAsocialGenerator)
    reg.generators.register("EventLogSocialPreChoiceGenerator", EventLogSocialPreChoiceGenerator)
    reg.generators.register("EventLogSocialPostOutcomeGenerator", EventLogSocialPostOutcomeGenerator)

    # environments + block runners
    reg.bandits.register("BernoulliBanditEnv", BernoulliBanditEnv)
    reg.tasks.register("BanditBlockRunner", BanditBlockRunner)
    reg.tasks.register("SocialBanditBlockRunner", SocialBanditBlockRunner)
