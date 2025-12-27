import json
import numpy as np
from scipy.special import softmax
from numpy.typing import NDArray
from collections.abc import Iterable
from generator import DynamicValueShapeModel
from mle import DynamicValueShapeMLE
from datetime import datetime
from pathlib import Path


class Other:
    def __init__(self, n_options: int, lr: float, beta: float):
        self.n_options = n_options
        self.values = np.full(n_options, 0.5, dtype=float)

        if lr < 0 or lr > 1:
            raise ValueError(
                f"The leraning rate should be between 0 and 1. {lr} is given."
            )
        self.lr = lr

        if beta < 0:
            raise ValueError(
                f"The inverse temperature shuold be positive. {beta} is given"
            )
        self.beta = beta

    def reset(self):
        self.values = np.full(self.n_options, 0.5, dtype=float)

    def learn_from_self(self, choice: int, reward: int):
        if choice >= self.n_options:
            raise ValueError(
                f"choice is out of range. It should be less than {self.n_options} but {choice} is given."
            )

        self.values[choice] = self.values[choice] + self.lr * (
            reward - self.values[choice]
        )

    def choose(self) -> None:
        prob = softmax(self.values * self.beta)
        return np.random.choice(self.n_options, p=prob)


def simulate_session(yourself, other, reward_probs: NDArray, n_trials: int) -> Iterable:
    choice_data = {
        "self_choices": [],
        "self_rewards": [],
        "other_choices": [],
        "other_rewards": [],
    }

    for t in range(n_trials):
        oc = other.choose()
        orw = int(reward_probs[oc] > np.random.rand())

        other.learn_from_self(oc, orw)
        yourself.learn_from_other(oc, orw)

        sc = yourself.choose()
        rw = int(reward_probs[sc] > np.random.rand())

        choice_data["self_choices"].append(sc)
        choice_data["self_rewards"].append(rw)
        choice_data["other_choices"].append(oc)
        choice_data["other_rewards"].append(orw)

    return choice_data


def simulate_exp(
    yourself, other, reward_probs: NDArray, n_trials: int, n_sessions: int
):
    data = []
    for s in range(n_sessions):
        yourself.reset()
        other.reset()
        choice_data = simulate_session(yourself, other, reward_probs, n_trials)
        data.append(choice_data)

    return data


def run_experiment(path):
    with open(path, "r") as f:
        setting = json.load(f)
    self_lr = float(setting["self_lr"])
    self_beta = float(setting["self_beta"])
    self_rel_coef = float(setting["self_rel_coef"])
    self_rel_const = float(setting["self_rel_const"])

    other_lr = float(setting["other_lr"])
    other_beta = float(setting["other_beta"])

    n_trials = int(setting["n_trials"])
    n_sessions = int(setting["n_sessions"])
    seed = int(setting["seed"])

    reward_probs = np.asarray(setting["reward_probs"], dtype=float)
    n_options = int(reward_probs.size)

    np.random.seed(seed)

    yourself = DynamicValueShapeModel(
        lr=self_lr,
        beta=self_beta,
        rel_coef=self_rel_coef,
        rel_const=self_rel_const,
        n_options=n_options,
    )
    other = Other(n_options, other_lr, other_beta)
    choice_data = simulate_exp(yourself, other, reward_probs, n_trials, n_sessions)

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    out_dir = Path(f"../results/param_rec/DynamicValueShapeModel/mle/{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DynamicValueShapeMLE(n_options)
    # Fit one shared parameter set across all sessions
    res = model.fit_mle(sessions=choice_data, seed=seed)

    def _json_default(o):
        # Handle numpy scalars/arrays gracefully
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    # Save setting to JSON
    setting_path = out_dir / "exp_setting.json"
    with open(setting_path, "w") as f:
        json.dump(setting, f, indet=2, sort_key=True, default=_json_default)

    # Save results to JSON
    res_path = out_dir / "mle_result.json"
    with open(res_path, "w") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=_json_default)

    return res


def main():
    exp_setting_path = Path("./experiment_setting.json")
    run_experiment(exp_setting_path)


if __name__ == "__main__":
    main()