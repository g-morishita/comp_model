import pytest


def _types(log):
    return [e.type.name for e in log.events]


def test_partner_self_row_parsing_and_orders():
    """Parses the user's example schema and correctly decodes hyphen-delimited orders."""
    from comp_model_core.events.convert import trials_from_partner_self_rows, PartnerSelfTrialTableColumns

    rows = [
        {
            "id": 1,
            "block": 2,
            "trial": 0,
            "partner_choice": 0,
            "partner_reward": 1,
            "partner_choice_pos_idx": 0.0,
            "self_choice": 0.0,
            "self_choice_pos_idx": 2.0,
            "rt": 1.1457259583257837,
            "obs_randomized_img_order": "0-1-2",
            "self_randomized_img_order": "2-1-0",
            "is_partner_precise": False,
        }
    ]
    cols = PartnerSelfTrialTableColumns()
    assert not hasattr(cols, "self_randomized_img_order")
    assert not hasattr(cols, "obs_randomized_img_order")
    assert not hasattr(cols, "is_partner_precise")


    trials = trials_from_partner_self_rows(rows)
    assert len(trials) == 1
    tr = trials[0]

    assert tr.t == 0
    assert tr.choice == 0
    # Social info packed as lists
    assert tr.others_choices == [0]
    assert tr.observed_others_outcomes == [1.0]

    # Orders become int lists
    assert tr.info["self_randomized_img_order"] == [2, 1, 0]
    assert tr.social_info["obs_randomized_img_order"] == [0, 1, 2]
    assert tr.social_info["is_partner_precise"] is False


def test_event_log_from_trials_timing_orders():
    """Event ordering matches the requested timing mode."""
    from comp_model_core.data.types import Trial
    from comp_model_core.events.convert import event_log_from_trials

    tr = Trial(
        t=0,
        state=0,
        choice=1,
        observed_outcome=0.5,
        outcome=0.5,
        available_actions=[0, 1],
        info={"rt": 1.2},
        others_choices=[0],
        observed_others_outcomes=[1.0],
        others_outcomes=None,
        social_info={"is_partner_precise": True},
    )

    log_asocial = event_log_from_trials(block_id="b", condition="c", trials=[tr], timing="asocial")
    assert _types(log_asocial) == ["BLOCK_START", "CHOICE", "OUTCOME"]

    log_pre = event_log_from_trials(block_id="b", condition="c", trials=[tr], timing="pre_choice")
    assert _types(log_pre) == ["BLOCK_START", "SOCIAL_OBSERVED", "CHOICE", "OUTCOME"]

    log_post = event_log_from_trials(block_id="b", condition="c", trials=[tr], timing="post_outcome")
    assert _types(log_post) == ["BLOCK_START", "CHOICE", "OUTCOME", "SOCIAL_OBSERVED"]

    # BLOCK_START payload must include condition (needed by replay likelihood)
    assert log_pre.events[0].payload["condition"] == "c"

    # idx is contiguous and starts at 0
    assert [e.idx for e in log_post.events] == list(range(len(log_post.events)))


def test_missing_self_outcomes_warn():
    """Missing self outcomes warn and remain None."""
    from comp_model_core.events.convert import event_log_from_partner_self_rows

    row = {
        "id": 1,
        "block": 1,
        "trial": 0,
        "self_choice": 1,
        # no self outcome columns
        "partner_choice": 0,
        "partner_reward": 1,
    }

    with pytest.warns(UserWarning, match="Self observed outcome is missing"):
        log_missing = event_log_from_partner_self_rows(
            [row],
            block_id="1",
            condition="cond",
            timing="asocial",
        )

    # OUTCOME event payload should have None if self outcomes are missing.
    assert log_missing.events[-1].type.name == "OUTCOME"
    assert log_missing.events[-1].payload["observed_outcome"] is None

    row_partial = {
        "id": 1,
        "block": 1,
        "trial": 1,
        "self_choice": 1,
        # self true outcome exists, but observed self outcome is missing
        "self_outcome": 0,
        "partner_choice": 0,
        "partner_reward": 1,
    }
    with pytest.warns(UserWarning, match="Self observed outcome is missing"):
        log_partial = event_log_from_partner_self_rows(
            [row_partial],
            block_id="1",
            condition="cond",
            timing="asocial",
        )
    assert log_partial.events[-1].payload["observed_outcome"] is None
    assert log_partial.events[-1].payload["outcome"] == 0.0


def test_merge_self_and_demo_rows_aggregates_multiple_demos():
    from comp_model_core.events.convert import merge_self_and_demo_rows

    self_rows = [
        {"id": 1, "block": 1, "trial": 0, "self_choice": 1},
    ]
    demo_rows = [
        {"id": 1, "block": 1, "trial": 0, "partner_choice": 0, "partner_reward": 1, "demo_id": "a"},
        {"id": 1, "block": 1, "trial": 0, "partner_choice": 2, "partner_reward": 0, "demo_id": "b"},
    ]

    merged = merge_self_and_demo_rows(
        self_rows=self_rows,
        demo_rows=demo_rows,
        keys=("id", "block", "trial"),
        demo_choice_col="partner_choice",
        demo_reward_col="partner_reward",
        demo_id_col="demo_id",
    )
    assert len(merged) == 1
    r = merged[0]
    assert r["others_choices"] == [0, 2]
    assert r["observed_others_outcomes"] == [1.0, 0.0]
    assert r["social_info"]["demo_ids"] == ["a", "b"]


def test_event_log_from_any_rows_infers_forms():
    from comp_model_core.events.convert import event_log_from_any_rows

    combined = [
        {"id": 1, "block": 1, "trial": 0, "self_choice": 1, "partner_choice": 0, "partner_reward": 1},
    ]
    log1 = event_log_from_any_rows(combined, block_id="1", condition="c", timing="pre_choice")
    assert _types(log1)[1] == "SOCIAL_OBSERVED"  # inferred combined form

    separate = {
        "self": [{"id": 1, "block": 1, "trial": 0, "self_choice": 1}],
        "demo": [{"id": 1, "block": 1, "trial": 0, "partner_choice": 0, "partner_reward": 1}],
    }
    log2 = event_log_from_any_rows(separate, block_id="1", condition="c", timing="pre_choice")
    assert _types(log2)[1] == "SOCIAL_OBSERVED"


def test_as_rows_accepts_dict_of_columns_and_pandas_like():
    """The ingestion layer accepts common tabular forms."""
    from comp_model_core.events.convert import trials_from_partner_self_rows, PartnerSelfTrialTableColumns

    # dict-of-columns
    dcols = {
        "id": [1, 1],
        "block": [1, 1],
        "trial": [0, 1],
        "self_choice": [0, 1],
        "partner_choice": [1, 0],
        "partner_reward": [1, 0],
    }
    trials = trials_from_partner_self_rows(dcols)
    assert [t.t for t in trials] == [0, 1]

    # pandas-like object
    class _DF:
        def __init__(self, rows):
            self._rows = rows

        def to_dict(self, orient="records"):
            assert orient in ("records",)
            return list(self._rows)

    df_like = _DF(
        [
            {"id": 1, "block": 1, "trial": 0, "self_choice": 0, "partner_choice": 1, "partner_reward": 1},
        ]
    )
    trials2 = trials_from_partner_self_rows(df_like)
    assert len(trials2) == 1


def test_invalid_timing_raises():
    from comp_model_core.events.convert import event_log_from_trials

    with pytest.raises(ValueError):
        event_log_from_trials(block_id="b", condition="c", trials=[], timing="bogus")


def test_social_event_emitted_with_empty_payload_when_missing():
    """For social timings, SOCIAL_OBSERVED is emitted even if trial has no social data."""
    from comp_model_core.data.types import Trial
    from comp_model_core.events.convert import event_log_from_trials

    tr = Trial(t=0, state=0, choice=0, observed_outcome=None, outcome=None)
    log = event_log_from_trials(block_id="b", condition="c", trials=[tr], timing="pre_choice")
    assert _types(log) == ["BLOCK_START", "SOCIAL_OBSERVED", "CHOICE", "OUTCOME"]
    payload = log.events[1].payload
    assert payload["others_choices"] == []
    assert payload["others_outcomes"] == []
    assert payload["observed_others_outcomes"] is None


def test_choice_none_omits_choice_and_outcome_but_keeps_social():
    """If choice is None, we omit CHOICE/OUTCOME but keep SOCIAL_OBSERVED for social timings."""
    from comp_model_core.data.types import Trial
    from comp_model_core.events.convert import event_log_from_trials

    tr = Trial(t=0, state=0, choice=None, observed_outcome=1.0, outcome=1.0, others_choices=[1])
    log = event_log_from_trials(block_id="b", condition="c", trials=[tr], timing="pre_choice")
    assert _types(log) == ["BLOCK_START", "SOCIAL_OBSERVED"]
