import numpy as np
from scipy.special import expit, softmax
from scipy.optimize import minimize
from numpy.typing import NDArray
from collections.abc import Iterable, Mapping
from typing import Any


class DynamicValueShapeMLE:
    """MLE utilities for DynamicValueShapeModel.

    It supports fitting a *single* shared parameter set across multiple sessions
    by summing log-likelihoods across sessions.
    """

    def __init__(self, n_options: int) -> None:
        self.n_options = int(n_options)

    @staticmethod
    def _validate_params(
        params: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        lr, beta, rel_coef, rel_const = map(float, params)
        if not (0.0 <= lr <= 1.0):
            raise ValueError(f"lr must be in [0, 1], got {lr}")
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        return lr, beta, rel_coef, rel_const

    def compute_ll_per_session(
        self,
        params: tuple[float, float, float, float],
        self_choices: NDArray,
        self_rewards: NDArray,
        other_choices: NDArray,
        other_rewards: NDArray,
    ) -> float:
        """Compute log-likelihood for one session under a fixed parameter set."""
        lr, beta, rel_coef, rel_const = self._validate_params(params)

        # Flatten inputs to 1D and check lengths match
        self_choices = np.asarray(self_choices).ravel()
        self_rewards = np.asarray(self_rewards).ravel()
        other_choices = np.asarray(other_choices).ravel()
        other_rewards = np.asarray(other_rewards).ravel()

        T = int(self_choices.shape[0])
        if not (
            int(self_rewards.shape[0]) == T
            and int(other_choices.shape[0]) == T
            and int(other_rewards.shape[0]) == T
        ):
            raise ValueError(
                "All inputs must have the same length. "
                f"Got lengths: self_choices={T}, self_rewards={self_rewards.shape[0]}, "
                f"other_choices={other_choices.shape[0]}, other_rewards={other_rewards.shape[0]}"
            )

        # ---- initialize per-session state ----
        values = np.full(self.n_options, 0.5, dtype=float)
        tracked_choices = np.ones(self.n_options, dtype=float)

        ll = 0.0
        eps = 1e-15  # avoid log(0)

        for t in range(T):
            # 1) Observe OTHER: update values based on other choice/outcome
            oc = int(other_choices[t])  # oc stands for other choice
            if oc < 0 or oc >= self.n_options:
                raise ValueError(
                    f"other_choices[{t}]={oc} out of range [0, {self.n_options - 1}]"
                )
            orw = float(other_rewards[t])  # orw stands for other reward

            expected_prob = tracked_choices[oc] / tracked_choices.sum()
            reliability = expit(rel_coef * expected_prob + rel_const)

            # shaping (toward 1) then RL update
            values[oc] = values[oc] + reliability * (1.0 - values[oc])
            values[oc] = values[oc] + lr * (orw - values[oc])
            tracked_choices[oc] += 1.0

            # 2) SELF choice likelihood under current values
            sc = int(self_choices[t])
            if sc < 0 or sc >= self.n_options:
                raise ValueError(
                    f"self_choices[{t}]={sc} out of range [0, {self.n_options - 1}]"
                )
            probs = softmax(values * beta)
            ll += float(np.log(probs[sc] + eps))

            # 3) Learn from SELF outcome
            srw = float(self_rewards[t])
            values[sc] = values[sc] + lr * (srw - values[sc])

        return ll

    @staticmethod
    def _as_session_tuple(session: Any) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Normalize a session into (self_choices, self_rewards, other_choices, other_rewards)."""
        if isinstance(session, Mapping):
            return (
                session["self_choices"],
                session["self_rewards"],
                session["other_choices"],
                session["other_rewards"],
            )
        if isinstance(session, (tuple, list)) and len(session) == 4:
            return (session[0], session[1], session[2], session[3])
        raise TypeError(
            "Each session must be either a dict with keys "
            "{'self_choices','self_rewards','other_choices','other_rewards'} "
            "or a 4-tuple/list (self_choices, self_rewards, other_choices, other_rewards)."
        )

    def compute_ll(
        self,
        params: tuple[float, float, float, float],
        sessions: Iterable[Any],
    ) -> float:
        """Sum log-likelihood across multiple sessions under a shared parameter set."""
        ll_total = 0.0
        for sess in sessions:
            sc, sr, oc, orw = self._as_session_tuple(sess)
            ll_total += self.compute_ll_per_session(
                params=params,
                self_choices=sc,
                self_rewards=sr,
                other_choices=oc,
                other_rewards=orw,
            )
        return float(ll_total)

    def neg_ll(
        self,
        params: tuple[float, float, float, float],
        sessions: Iterable[Any],
    ) -> float:
        """Negative summed log-likelihood across sessions."""
        return -self.compute_ll(params=params, sessions=sessions)

    def fit_mle(
        self,
        sessions: Iterable[Any],
        x0: tuple[float, float, float, float] | None = None,
        n_restarts: int = 5,
        seed: int | None = None,
    ) -> dict:
        """Estimate (lr, beta, rel_coef, rel_const) by maximizing summed log-likelihood.

        Uses L-BFGS-B to minimize the negative log-likelihood.

        Parameters
        ---------
        sessions:
            Iterable of sessions. Each session can be either:
              - dict with keys: self_choices, self_rewards, other_choices, other_rewards
              - 4-tuple/list: (self_choices, self_rewards, other_choices, other_rewards)
        x0:
            Optional initial guess (lr, beta, rel_coef, rel_const).
        n_restarts:
            Number of random restarts (including x0 if provided).
        seed:
            RNG seed for restarts.

        Returns
        -------
        dict with keys:
          - params: dict of MLE estimates
          - ll: maximized summed log-likelihood
          - success: optimizer success flag
          - message: optimizer message
          - n_restarts: number of restarts attempted
        """
        rng = np.random.default_rng(seed)

        # Materialize sessions once (so we can iterate multiple times in restarts)
        sessions_list = list(sessions)
        if len(sessions_list) == 0:
            raise ValueError("sessions must contain at least one session")

        # bounds: lr in [0,1]; beta >= 0; rel_* unbounded
        bounds = [(0.0, 1.0), (0.0, None), (None, None), (None, None)]

        def _random_x0() -> tuple[float, float, float, float]:
            lr0 = float(rng.uniform(0.05, 0.95))
            beta0 = float(rng.uniform(0.1, 10.0))
            rel_coef0 = float(rng.normal(0.0, 2.0))
            rel_const0 = float(rng.normal(0.0, 2.0))
            return (lr0, beta0, rel_coef0, rel_const0)

        x0_list: list[tuple[float, float, float, float]] = []
        if x0 is not None:
            x0_list.append(tuple(map(float, x0)))
        for _ in range(max(0, int(n_restarts) - len(x0_list))):
            x0_list.append(_random_x0())

        best_res = None
        best_nll = float("inf")

        def _objective(p: NDArray) -> float:
            return self.neg_ll(
                params=(float(p[0]), float(p[1]), float(p[2]), float(p[3])),
                sessions=sessions_list,
            )

        for x0_try in x0_list:
            res = minimize(
                fun=_objective,
                x0=np.asarray(x0_try, dtype=float),
                method="L-BFGS-B",
                bounds=bounds,
            )
            nll = float(res.fun)
            if nll < best_nll:
                best_nll = nll
                best_res = res

        assert best_res is not None
        lr_hat, beta_hat, rel_coef_hat, rel_const_hat = map(float, best_res.x)
        ll_hat = -best_nll

        return {
            "params": {
                "lr": lr_hat,
                "beta": beta_hat,
                "rel_coef": rel_coef_hat,
                "rel_const": rel_const_hat,
            },
            "ll": ll_hat,
            "success": bool(best_res.success),
            "message": str(best_res.message),
            "n_restarts": len(x0_list),
        }


def assert_raises(exc_type, fn, *args, **kwargs):
    """Tiny helper: assert a function raises exc_type."""
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(
            f"Expected {exc_type.__name__}, but got {type(e).__name__}: {e}"
        ) from e
    raise AssertionError(
        f"Expected {exc_type.__name__} to be raised, but no exception was raised."
    )


def assert_allclose(a, b, tol=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not np.allclose(a, b, atol=tol, rtol=0):
        raise AssertionError(f"Not close.\nGot: {a}\nExp: {b}")


if __name__ == "__main__":
    # ---- Smoke test: instantiate ----
    mle = DynamicValueShapeMLE(n_options=3)
    assert mle.n_options == 3

    # ---- Test: parameter validation ----
    dummy = np.array([0, 1, 2])
    assert_raises(
        ValueError,
        mle.compute_ll_per_session,
        (-0.1, 1.0, 0.0, 0.0),
        dummy,
        dummy,
        dummy,
        dummy,
    )
    assert_raises(
        ValueError,
        mle.compute_ll_per_session,
        (1.1, 1.0, 0.0, 0.0),
        dummy,
        dummy,
        dummy,
        dummy,
    )
    assert_raises(
        ValueError,
        mle.compute_ll_per_session,
        (0.5, -1.0, 0.0, 0.0),
        dummy,
        dummy,
        dummy,
        dummy,
    )

    # ---- Test: length mismatch ----
    sc = np.array([0, 1, 2])
    sr = np.array([1, 0, 1])
    oc = np.array([0, 1])
    orw = np.array([1, 1, 1])
    assert_raises(
        ValueError,
        mle.compute_ll_per_session,
        (0.2, 1.0, 0.0, 0.0),
        sc,
        sr,
        oc,
        orw,
    )

    # ---- Test: out-of-range choice indices ----
    sc = np.array([0, 3])
    sr = np.array([1, 1])
    oc = np.array([0, 1])
    orw = np.array([1, 1])
    assert_raises(
        ValueError,
        mle.compute_ll_per_session,
        (0.2, 1.0, 0.0, 0.0),
        sc,
        sr,
        oc,
        orw,
    )

    sc = np.array([0, 1])
    sr = np.array([1, 1])
    oc = np.array([0, 3])
    orw = np.array([1, 1])
    assert_raises(
        ValueError,
        mle.compute_ll_per_session,
        (0.2, 1.0, 0.0, 0.0),
        sc,
        sr,
        oc,
        orw,
    )

    # ---- Test: likelihood calculation matches manual computation for a tiny session ----
    # Setup a 2-trial session, 3 options.
    # rel_coef=0, rel_const=0 => reliability = sigmoid(0)=0.5 always
    params = (0.2, 1.5, 0.0, 0.0)  # lr, beta, rel_coef, rel_const
    sc = np.array([0, 0])
    sr = np.array([1.0, 0.0])
    oc = np.array([1, 1])
    orw = np.array([1.0, 1.0])

    # Manual forward pass
    lr, beta, rel_coef, rel_const = params
    values = np.full(3, 0.5, dtype=float)
    tracked = np.ones(3, dtype=float)
    ll_manual = 0.0
    eps = 1e-15

    for t in range(2):
        # other update
        o = int(oc[t])
        expected_prob = tracked[o] / tracked.sum()
        reliability = expit(rel_coef * expected_prob + rel_const)  # 0.5
        values[o] = values[o] + reliability * (1.0 - values[o])
        values[o] = values[o] + lr * (float(orw[t]) - values[o])
        tracked[o] += 1.0

        # self choice ll
        probs = softmax(values * beta)
        ll_manual += float(np.log(probs[int(sc[t])] + eps))

        # self update
        values[int(sc[t])] = values[int(sc[t])] + lr * (
            float(sr[t]) - values[int(sc[t])]
        )

    ll_code = mle.compute_ll_per_session(params, sc, sr, oc, orw)
    assert_allclose(ll_code, ll_manual)

    # ---- Test: likelihood calculation matches manual computation when reliability depends on tracked choices ----
    # Here we choose rel_coef/rel_const so reliability changes across trials because expected_prob changes.
    params2 = (0.3, 2.0, 3.0, -1.0)  # lr, beta, rel_coef, rel_const
    sc2 = np.array([2, 2, 0])
    sr2 = np.array([0.0, 1.0, 1.0])
    oc2 = np.array([1, 1, 1])
    orw2 = np.array([1.0, 0.0, 1.0])

    lr2, beta2, rel_coef2, rel_const2 = params2
    values2 = np.full(3, 0.5, dtype=float)
    tracked2 = np.ones(3, dtype=float)
    ll_manual2 = 0.0

    reliabilities = []

    for t in range(3):
        o = int(oc2[t])
        expected_prob2 = tracked2[o] / tracked2.sum()
        reliability2 = expit(rel_coef2 * expected_prob2 + rel_const2)
        reliabilities.append(reliability2)

        values2[o] = values2[o] + reliability2 * (1.0 - values2[o])
        values2[o] = values2[o] + lr2 * (float(orw2[t]) - values2[o])
        tracked2[o] += 1.0

        probs2 = softmax(values2 * beta2)
        ll_manual2 += float(np.log(probs2[int(sc2[t])] + eps))

        values2[int(sc2[t])] = values2[int(sc2[t])] + lr2 * (
            float(sr2[t]) - values2[int(sc2[t])]
        )

    # Reliability should not be constant across these trials
    assert not np.isclose(reliabilities[0], reliabilities[1]) or not np.isclose(
        reliabilities[1], reliabilities[2]
    )

    ll_code2 = mle.compute_ll_per_session(params2, sc2, sr2, oc2, orw2)
    assert_allclose(ll_code2, ll_manual2)

    # ---- Test: multi-session LL equals sum of per-session LL ----
    sess1 = (sc, sr, oc, orw)
    sess2 = (np.array([2]), np.array([1.0]), np.array([0]), np.array([0.0]))
    ll_sum = mle.compute_ll_per_session(params, *sess1) + mle.compute_ll_per_session(
        params, *sess2
    )
    ll_multi = mle.compute_ll(params=params, sessions=[sess1, sess2])
    assert_allclose(ll_multi, ll_sum)

    # Also allow dict sessions
    sess2d = {
        "self_choices": sess2[0],
        "self_rewards": sess2[1],
        "other_choices": sess2[2],
        "other_rewards": sess2[3],
    }
    ll_multi2 = mle.compute_ll(params=params, sessions=[sess1, sess2d])
    assert_allclose(ll_multi2, ll_sum)

    # ---- Test: fit_mle returns required keys and finite ll (smoke) ----
    # Note: we don't assert recovery accuracy here because optimization can depend on data/initialization.
    res = mle.fit_mle(sessions=[sess1, sess2], n_restarts=3, seed=0)
    assert isinstance(res, dict)
    assert set(res.keys()) == {"params", "ll", "success", "message", "n_restarts"}
    assert set(res["params"].keys()) == {"lr", "beta", "rel_coef", "rel_const"}
    assert np.isfinite(res["ll"])
    assert res["n_restarts"] == 3

    # ---- Test: fit_mle is consistent across session representations (tuple vs dict) ----
    res_tuple = mle.fit_mle(sessions=[sess1], n_restarts=2, seed=1)
    sess1d = {
        "self_choices": sess1[0],
        "self_rewards": sess1[1],
        "other_choices": sess1[2],
        "other_rewards": sess1[3],
    }
    res_dict = mle.fit_mle(sessions=[sess1d], n_restarts=2, seed=1)
    assert_allclose(res_tuple["ll"], res_dict["ll"], tol=1e-9)

    print("All inline tests passed ✅")
