"""Generators that emit explicit event logs.

The generators in this module simulate subjects and store a detailed
:class:`comp_model_core.events.types.EventLog` on each block for later replay.
Event logs are consumed by the event-log likelihood and Stan exporters.

Notes
-----
The event stream specifies when model resets, social observations, choices, and
outcome updates occur. This timing is critical for likelihood replay in
:mod:`comp_model_impl.likelihood.event_log_replay` and for Stan data export in
:mod:`comp_model_impl.estimators.stan.exporter`.

Examples
--------
Generate an asocial subject with explicit event logs:

>>> import numpy as np
>>> from comp_model_impl.generators.event_log import EventLogAsocialGenerator
>>> from comp_model_impl.models import QRL
>>> from comp_model_core.plans.block import BlockPlan
>>> from comp_model_impl.register import make_registry
>>> from comp_model_impl.tasks.build import build_runner_for_plan
>>> plan = BlockPlan(
...     block_id="b1",
...     n_trials=2,
...     condition="c1",
...     bandit_type="BernoulliBanditEnv",
...     bandit_config={"probs": [0.2, 0.8]},
...     trial_specs=[{"self_outcome": {"kind": "VERIDICAL"}, "available_actions": [0, 1]}] * 2,
... )
>>> reg = make_registry()
>>> builder = lambda p: build_runner_for_plan(plan=p, registries=reg)
>>> gen = EventLogAsocialGenerator()
>>> subj = gen.simulate_subject(
...     subject_id="s1",
...     block_runner_builder=builder,
...     model=QRL(),
...     params={"alpha": 0.2, "beta": 3.0},
...     block_plans=[plan],
...     rng=np.random.default_rng(0),
... )
>>> subj.blocks[0].event_log is not None
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, cast

import numpy as np

from comp_model_core.data.types import Trial, Block, SubjectData
from comp_model_core.errors import CompatibilityError
from comp_model_core.interfaces.bandit import BanditEnv, SocialBanditEnv
from comp_model_core.interfaces.generator import Generator
from comp_model_core.interfaces.block_runner import BlockRunner, SocialBlockRunner
from comp_model_core.interfaces.model import ComputationalModel, SocialComputationalModel
from comp_model_core.plans.block import BlockPlan
from comp_model_core.events.types import Event, EventLog, EventType
from comp_model_core.validation import validate_block_plan

BlockRunnerBuilder = Callable[[BlockPlan], BlockRunner]

_EPS = 1e-12


def _ensure_model_supports(model: ComputationalModel, bandit: BanditEnv) -> None:
    """Validate that the model supports the bandit environment.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Model to validate.
    bandit : comp_model_core.interfaces.bandit.BanditEnv
        Environment instance providing a spec.

    Raises
    ------
    comp_model_core.errors.CompatibilityError
        If the model does not support the environment spec.

    See Also
    --------
    EventLogAsocialGenerator.simulate_subject
    EventLogSocialPreChoiceGenerator.simulate_subject
    EventLogSocialPostOutcomeGenerator.simulate_subject
    """
    if not model.supports(spec=bandit.spec):
        raise CompatibilityError("The computational model is not compatible with the current task.")


def _reset_block(model: ComputationalModel, bandit: BanditEnv, rng: np.random.Generator) -> None:
    """Reset bandit and model state for a new block.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Model to reset.
    bandit : comp_model_core.interfaces.bandit.BanditEnv
        Environment to reset.
    rng : numpy.random.Generator
        RNG used by the environment reset.

    See Also
    --------
    comp_model_core.interfaces.block_runner.BlockRunner.reset
    comp_model_core.interfaces.model.ComputationalModel.reset_block
    """
    bandit.reset(rng=rng)
    model.reset_block(spec=bandit.spec)


def _maybe_set_condition(model: ComputationalModel, condition: str) -> None:
    """Set block-level condition if the model supports it.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Model that may expose a ``set_condition`` method.
    condition : str
        Condition label to set.

    See Also
    --------
    comp_model_impl.models.within_subject_shared_delta.ConditionedSharedDeltaModel.set_condition
    """
    setter = getattr(model, "set_condition", None)
    if callable(setter):
        setter(str(condition))


def _requirements_for_model(model: ComputationalModel):
    """Return plan-validation requirements, unwrapping conditioned wrappers.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Model or wrapper.

    Returns
    -------
    tuple
        Requirements declared by the base model when wrapped, otherwise by the model.

    See Also
    --------
    comp_model_core.validation.validate_block_plan
    comp_model_impl.models.within_subject_shared_delta.ConditionedSharedDeltaModel
    """
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "requirements"):
        return base.requirements()  # type: ignore[no-any-return]
    return model.requirements()


def _build_event_log(events: list[Event], *, metadata: dict) -> EventLog:
    """Construct an :class:`EventLog` from events and metadata.

    Parameters
    ----------
    events : list[comp_model_core.events.types.Event]
        Event list in time order.
    metadata : dict
        Metadata to attach to the log.

    Returns
    -------
    comp_model_core.events.types.EventLog
        Event log wrapper.

    See Also
    --------
    comp_model_core.events.types.EventLog
    """
    return EventLog(events=events, metadata=metadata)


def _mask_and_renorm(probs: np.ndarray, available_actions: Sequence[int] | None) -> np.ndarray:
    """Mask action probabilities and renormalize.

    Parameters
    ----------
    probs : numpy.ndarray
        Raw action probability vector.
    available_actions : Sequence[int] or None
        If provided, only these actions are allowed and will retain mass.

    Returns
    -------
    numpy.ndarray
        Masked and renormalized probability vector.

    Examples
    --------
    >>> import numpy as np
    >>> _mask_and_renorm(np.array([0.2, 0.3, 0.5]), available_actions=[0, 2])
    array([0.28571429, 0.        , 0.71428571])

    See Also
    --------
    comp_model_impl.likelihood.event_log_replay._mask_and_renorm
    """
    p = np.asarray(probs, dtype=float).copy()
    if available_actions is None:
        return p

    mask = np.zeros_like(p, dtype=bool)
    for a in available_actions:
        mask[int(a)] = True
    p[~mask] = 0.0
    s = float(p.sum())
    return p / max(s, _EPS)


@dataclass(slots=True)
class EventLogAsocialGenerator(Generator):
    """
    Asocial simulation with explicit event log.

    Per block:
      BLOCK_START (signals reset)
      For each trial:
        CHOICE -> OUTCOME

    The event log is stored in Block.event_log.
    """

    def simulate_subject(
        self,
        *,
        subject_id: str,
        block_runner_builder: BlockRunnerBuilder,
        model: ComputationalModel,
        params: Mapping[str, float],
        block_plans: Sequence[BlockPlan],
        rng: np.random.Generator,
    ) -> SubjectData:
        """Simulate an asocial subject and return event-log data.

        Parameters
        ----------
        subject_id : str
            Subject identifier.
        block_runner_builder : Callable[[BlockPlan], BlockRunner]
            Factory that creates a block runner for each plan.
        model : comp_model_core.interfaces.model.ComputationalModel
            Model used to generate choices.
        params : Mapping[str, float]
            Parameter values to set on the model before simulation.
        block_plans : Sequence[comp_model_core.plans.block.BlockPlan]
            Block plans to simulate.
        rng : numpy.random.Generator
            RNG used for stochastic choices and environments.

        Returns
        -------
        comp_model_core.data.types.SubjectData
            Subject data with event logs attached to each block.

        Notes
        -----
        The event order is ``BLOCK_START -> (CHOICE -> OUTCOME)*`` and is used
        by the event-log likelihood in :mod:`comp_model_impl.likelihood.event_log_replay`.

        Examples
        --------
        >>> # gen = EventLogAsocialGenerator()
        >>> # subject = gen.simulate_subject(...)  # doctest: +SKIP
        """
        model.set_params(params)

        blocks: list[Block] = []

        for plan in block_plans:
            runner = block_runner_builder(plan)
            spec = runner.spec

            if getattr(spec, "is_social", False):
                raise CompatibilityError("EventLogAsocialGenerator cannot run a social task (spec.is_social=True).")

            _ensure_model_supports(model, runner)
            validate_block_plan(plan=plan, env_spec=spec, requirements=_requirements_for_model(model))

            # Within-subject designs: condition is block-level.
            _maybe_set_condition(model, plan.condition)
            _reset_block(model, runner, rng)

            trials: list[Trial] = []
            events: list[Event] = []

            # Mandatory: block reset marker
            events.append(
                Event(
                    idx=0,
                    type=EventType.BLOCK_START,
                    t=None,
                    state=None,
                    payload={"block_id": plan.block_id, "condition": plan.condition},
                )
            )
            idx = 1

            for t in range(int(plan.n_trials)):
                state = runner.get_state()
                ts = runner.trial_spec(t=t)
                aa = ts.available_actions

                probs = model.action_probs(state=state, spec=spec)
                probs = _mask_and_renorm(probs, aa)

                action = int(rng.choice(spec.n_actions, p=probs))
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.CHOICE,
                        t=t,
                        state=state,
                        payload={
                            "choice": action,
                            "available_actions": None if aa is None else list(aa),
                        },
                    )
                )
                idx += 1

                step = runner.step(t=t, action=action, rng=rng)

                events.append(
                    Event(
                        idx=idx,
                        type=EventType.OUTCOME,
                        t=t,
                        state=state,
                        payload={
                            "action": action,
                            "observed_outcome": step.observed_outcome,
                            "outcome": step.outcome,
                            "info": step.info or {},
                        },
                    )
                )
                idx += 1

                model.update(state=state, action=action, outcome=step.observed_outcome, spec=spec, info=step.info)

                trials.append(
                    Trial(
                        t=t,
                        state=state,
                        choice=action,
                        observed_outcome=step.observed_outcome,
                        outcome=step.outcome,
                        available_actions=None if aa is None else list(aa),
                        info=step.info or {},
                        others_choices=None,
                        others_outcomes=None,
                        observed_others_outcomes=None,
                        social_info={},
                    )
                )

            blocks.append(
                Block(
                    block_id=plan.block_id,
                    condition=plan.condition,
                    trials=trials,
                    env_spec=spec,
                    event_log=_build_event_log(events, metadata={"timing": "asocial"}),
                    metadata={
                        "plan": dict(plan.metadata),
                    },
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})


@dataclass(slots=True)
class EventLogSocialPreChoiceGenerator(Generator):
    """
    Social simulation with event log: SOCIAL_OBSERVED happens BEFORE CHOICE.

    Per block:
      BLOCK_START
      For each trial:
        SOCIAL_OBSERVED -> CHOICE -> OUTCOME
    """

    def simulate_subject(
        self,
        *,
        subject_id: str,
        block_runner_builder: BlockRunnerBuilder,
        model: ComputationalModel,
        params: Mapping[str, float],
        block_plans: Sequence[BlockPlan],
        rng: np.random.Generator,
    ) -> SubjectData:
        """Simulate a social subject with SOCIAL_OBSERVED before CHOICE.

        Parameters
        ----------
        subject_id : str
            Subject identifier.
        block_runner_builder : Callable[[BlockPlan], BlockRunner]
            Factory that creates a block runner for each plan.
        model : comp_model_core.interfaces.model.SocialComputationalModel
            Social model used to generate choices.
        params : Mapping[str, float]
            Parameter values to set on the model before simulation.
        block_plans : Sequence[comp_model_core.plans.block.BlockPlan]
            Block plans to simulate.
        rng : numpy.random.Generator
            RNG used for stochastic choices and environments.

        Returns
        -------
        comp_model_core.data.types.SubjectData
            Subject data with event logs attached to each block.

        Notes
        -----
        The event order is ``BLOCK_START -> SOCIAL_OBSERVED -> CHOICE -> OUTCOME``.
        This timing matches models that observe demonstrations before choosing.

        Examples
        --------
        >>> # gen = EventLogSocialPreChoiceGenerator()
        >>> # subject = gen.simulate_subject(...)  # doctest: +SKIP
        """
        model.set_params(params)

        blocks: list[Block] = []

        for plan in block_plans:
            runner = block_runner_builder(plan)
            spec = runner.spec

            if not spec.is_social:
                raise CompatibilityError("EventLogSocialPreChoiceGenerator requires spec.is_social=True.")
            if not isinstance(runner, SocialBlockRunner):
                raise CompatibilityError("Social blocks require a SocialBlockRunner runtime object.")
            if not isinstance(model, SocialComputationalModel):
                raise CompatibilityError("Social task requires a SocialComputationalModel.")


            _ensure_model_supports(model, runner)
            validate_block_plan(plan=plan, env_spec=spec, requirements=_requirements_for_model(model))

            _maybe_set_condition(model, plan.condition)
            _reset_block(model, runner, rng)

            sb = cast(SocialBlockRunner, runner)
            sm = cast(SocialComputationalModel, model)

            trials: list[Trial] = []
            events: list[Event] = []

            events.append(
                Event(
                    idx=0,
                    type=EventType.BLOCK_START,
                    t=None,
                    state=None,
                    payload={"block_id": plan.block_id, "condition": plan.condition},
                )
            )
            idx = 1

            for t in range(int(plan.n_trials)):
                state = sb.get_state()

                obs = sb.observe_others(t=t, rng=rng)
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.SOCIAL_OBSERVED,
                        t=t,
                        state=state,
                        payload={
                            "others_choices": list(obs.others_choices or []),
                            "others_outcomes": list(obs.others_outcomes or []),
                            "observed_others_outcomes": None if obs.observed_others_outcomes is None else list(obs.observed_others_outcomes),
                            "social_info": obs.info or {},
                        },
                    )
                )
                idx += 1
                sm.social_update(state=state, social=obs, spec=spec, info=None)

                ts = sb.trial_spec(t=t)
                aa = ts.available_actions

                probs = sm.action_probs(state=state, spec=spec)
                probs = _mask_and_renorm(probs, aa)

                action = int(rng.choice(spec.n_actions, p=probs))
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.CHOICE,
                        t=t,
                        state=state,
                        payload={"choice": action, "available_actions": None if aa is None else list(aa)},
                    )
                )
                idx += 1

                step = sb.step(t=t, action=action, rng=rng)
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.OUTCOME,
                        t=t,
                        state=state,
                        payload={
                            "action": action,
                            "observed_outcome": step.observed_outcome,
                            "outcome": step.outcome,
                            "info": step.info or {},
                        },
                    )
                )
                idx += 1

                sm.update(state=state, action=action, outcome=step.observed_outcome, spec=spec, info=step.info)

                trials.append(
                    Trial(
                        t=t,
                        state=state,
                        choice=action,
                        observed_outcome=step.observed_outcome,
                        outcome=step.outcome,
                        available_actions=None if aa is None else list(aa),
                        info=step.info or {},
                        others_choices=obs.others_choices,
                        others_outcomes=obs.others_outcomes,
                        observed_others_outcomes=obs.observed_others_outcomes,
                        social_info=obs.info or {},
                    )
                )

            blocks.append(
                Block(
                    block_id=plan.block_id,
                    condition=plan.condition,
                    trials=trials,
                    env_spec=spec,
                    event_log=_build_event_log(events, metadata={"timing": "pre_choice"}),
                    metadata={
                        "plan": dict(plan.metadata),
                    },
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})


@dataclass(slots=True)
class EventLogSocialPostOutcomeGenerator(Generator):
    """
    Social simulation with event log: SOCIAL_OBSERVED happens AFTER OUTCOME.

    Per block:
      BLOCK_START
      For each trial:
        CHOICE -> OUTCOME -> SOCIAL_OBSERVED
    """

    def simulate_subject(
        self,
        *,
        subject_id: str,
        block_runner_builder: BlockRunnerBuilder,
        model: ComputationalModel,
        params: Mapping[str, float],
        block_plans: Sequence[BlockPlan],
        rng: np.random.Generator,
    ) -> SubjectData:
        """Simulate a social subject with SOCIAL_OBSERVED after OUTCOME.

        Parameters
        ----------
        subject_id : str
            Subject identifier.
        block_runner_builder : Callable[[BlockPlan], BlockRunner]
            Factory that creates a block runner for each plan.
        model : comp_model_core.interfaces.model.SocialComputationalModel
            Social model used to generate choices.
        params : Mapping[str, float]
            Parameter values to set on the model before simulation.
        block_plans : Sequence[comp_model_core.plans.block.BlockPlan]
            Block plans to simulate.
        rng : numpy.random.Generator
            RNG used for stochastic choices and environments.

        Returns
        -------
        comp_model_core.data.types.SubjectData
            Subject data with event logs attached to each block.

        Notes
        -----
        The event order is ``BLOCK_START -> CHOICE -> OUTCOME -> SOCIAL_OBSERVED``.
        This timing matches models that learn from demonstrations after outcomes.

        Examples
        --------
        >>> # gen = EventLogSocialPostOutcomeGenerator()
        >>> # subject = gen.simulate_subject(...)  # doctest: +SKIP
        """
        model.set_params(params)

        blocks: list[Block] = []

        for plan in block_plans:
            runner = block_runner_builder(plan)
            spec = runner.spec

            if not spec.is_social:
                raise CompatibilityError("EventLogSocialPostOutcomeGenerator requires spec.is_social=True.")
            if not isinstance(runner, SocialBlockRunner):
                raise CompatibilityError("Social blocks require a SocialBlockRunner runtime object.")
            if not isinstance(model, SocialComputationalModel):
                raise CompatibilityError("Social task requires a SocialComputationalModel.")

            _ensure_model_supports(model, runner)
            validate_block_plan(plan=plan, env_spec=spec, requirements=_requirements_for_model(model))

            _maybe_set_condition(model, plan.condition)
            _reset_block(model, runner, rng)

            sb = cast(SocialBlockRunner, runner)
            sm = cast(SocialComputationalModel, model)

            trials: list[Trial] = []
            events: list[Event] = []

            events.append(
                Event(
                    idx=0,
                    type=EventType.BLOCK_START,
                    t=None,
                    state=None,
                    payload={"block_id": plan.block_id, "condition": plan.condition},
                )
            )
            idx = 1

            for t in range(int(plan.n_trials)):
                state = sb.get_state()
                ts = sb.trial_spec(t=t)
                aa = ts.available_actions

                probs = sm.action_probs(state=state, spec=spec)
                probs = _mask_and_renorm(probs, aa)

                action = int(rng.choice(spec.n_actions, p=probs))
                events.append(Event(idx=idx, type=EventType.CHOICE, t=t, state=state, payload={"choice": action, "available_actions": None if aa is None else list(aa)}))
                idx += 1

                step = sb.step(t=t, action=action, rng=rng)
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.OUTCOME,
                        t=t,
                        state=state,
                        payload={
                            "action": action,
                            "observed_outcome": step.observed_outcome,
                            "outcome": step.outcome,
                            "info": step.info or {},
                        },
                    )
                )
                idx += 1

                sm.update(state=state, action=action, outcome=step.observed_outcome, spec=spec, info=step.info)

                obs = sb.observe_others(t=t, rng=rng)
                events.append(
                    Event(
                        idx=idx,
                        type=EventType.SOCIAL_OBSERVED,
                        t=t,
                        state=state,
                        payload={
                            "others_choices": list(obs.others_choices or []),
                            "others_outcomes": list(obs.others_outcomes or []),
                            "observed_others_outcomes": None if obs.observed_others_outcomes is None else list(obs.observed_others_outcomes),
                            "social_info": obs.info or {},
                        },
                    )
                )
                idx += 1

                sm.social_update(state=state, social=obs, spec=spec, info=None)

                trials.append(
                    Trial(
                        t=t,
                        state=state,
                        choice=action,
                        observed_outcome=step.observed_outcome,
                        outcome=step.outcome,
                        available_actions=None if aa is None else list(aa),
                        info=step.info or {},
                        others_choices=obs.others_choices,
                        others_outcomes=obs.others_outcomes,
                        observed_others_outcomes=obs.observed_others_outcomes,
                        social_info=obs.info or {},
                    )
                )

            blocks.append(
                Block(
                    block_id=plan.block_id,
                    condition=plan.condition,
                    trials=trials,
                    env_spec=spec,
                    event_log=_build_event_log(events, metadata={"timing": "post_outcome"}),
                    metadata={
                        "plan": dict(plan.metadata),
                    },
                )
            )

        return SubjectData(subject_id=subject_id, blocks=blocks, metadata={})
