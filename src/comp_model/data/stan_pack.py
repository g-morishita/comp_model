from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .types import StudyData


@dataclass(frozen=True, slots=True)
class StanPacked:
    """Convenience wrapper if you want both data dict + index maps."""
    data: Dict[str, Any]
    subject_ids: list[str]
    block_ids: list[tuple[str, str]]  # (subject_id, block_id)


def pack_stan_data(study: StudyData) -> StanPacked:
    """
    Packs variable-length (subject, block, trial) into ragged arrays using indexing.

    We create:
      S = #subjects
      B = #blocks total
      T = #trials total

      subj_of_block[b] in 1..S
      start_t[b], end_t[b] in 1..T  (inclusive bounds)

      choice[t], reward[t], state[t]
      (optional) is_missing_choice[t], is_missing_reward[t]

    This is a common pattern for Stan with ragged sequences.
    """
    subject_ids = [s.subject_id for s in study.subjects]
    subj_index = {sid: i + 1 for i, sid in enumerate(subject_ids)}  # Stan 1-indexed

    block_ids: list[tuple[str, str]] = []
    subj_of_block: list[int] = []
    start_t: list[int] = []
    end_t: list[int] = []

    choice: list[int] = []
    reward: list[float] = []
    state: list[int] = []
    miss_choice: list[int] = []
    miss_reward: list[int] = []

    t_counter = 0
    for subj in study.subjects:
        for blk in subj.blocks:
            block_ids.append((subj.subject_id, blk.block_id))
            subj_of_block.append(subj_index[subj.subject_id])

            if len(blk.trials) == 0:
                # allow empty blocks (rare); represent as start=end+1
                start_t.append(t_counter + 1)
                end_t.append(t_counter)
                continue

            start_t.append(t_counter + 1)
            for tr in blk.trials:
                t_counter += 1
                state.append(int(tr.state))

                if tr.choice is None:
                    choice.append(1)       # dummy
                    miss_choice.append(1)
                else:
                    choice.append(int(tr.choice) + 1)  # to 1..A
                    miss_choice.append(0)

                if tr.reward is None:
                    reward.append(0.0)     # dummy
                    miss_reward.append(1)
                else:
                    reward.append(float(tr.reward))
                    miss_reward.append(0)

            end_t.append(t_counter)

    A = study.task_spec.n_actions
    data = {
        "S": len(subject_ids),
        "B": len(block_ids),
        "T": t_counter,
        "A": A,
        "subj_of_block": np.array(subj_of_block, dtype=int),
        "start_t": np.array(start_t, dtype=int),
        "end_t": np.array(end_t, dtype=int),
        "state": np.array(state, dtype=int),
        "choice": np.array(choice, dtype=int),
        "reward": np.array(reward, dtype=float),
        "miss_choice": np.array(miss_choice, dtype=int),
        "miss_reward": np.array(miss_reward, dtype=int),
    }
    return StanPacked(data=data, subject_ids=subject_ids, block_ids=block_ids)
