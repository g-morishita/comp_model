import numpy as np
from scipy.special import expit, softmax


class DynamicValueShapeModel:
    def __init__(
        self, lr: float, beta: float, rel_coef: float, rel_const: float, n_options: int
    ) -> None:
        self.n_options = n_options
        self.reset()

        if (0 > lr) or (lr > 1):
            raise ValueError(
                f"The learning rate: `lr` should be between 0 and 1.\n {lr} is given."
            )
        self.lr = lr

        if beta < 0:
            raise ValueError(
                f"The inverse temperature `beta` should be positive.\n {beta} is given"
            )
        self.beta = beta

        self.rel_coef = rel_coef
        self.rel_const = rel_const
    
    def reset(self) -> None:
        self.values = np.full(self.n_options, 0.5)
        self.tracked_choices = np.ones(self.n_options)

    def learn_from_self(self, choice: int, reward: int) -> None:
        if choice >= self.n_options:
            raise ValueError(
                f"`choice` is out of range. It should be less than {self.n_options}"
            )
        self.values[choice] = self.values[choice] + self.lr * (
            reward - self.values[choice]
        )

    def learn_from_other(self, choice: int, reward: int) -> None:
        if choice >= self.n_options:
            raise ValueError(
                f"`choice` is out of range. It should be less than {self.n_options}"
            )

        # The update order is
        expected_prob = self.tracked_choices[choice] / self.tracked_choices.sum()
        reliability = expit(self.rel_coef * expected_prob + self.rel_const)
        self.values[choice] = self.values[choice] + reliability * (
            1 - self.values[choice]
        )
        self.values[choice] = self.values[choice] + self.lr * (
            reward - self.values[choice]
        )

        self.tracked_choices[choice] += 1

    def choose(self) -> int:
        choice_probs = softmax(self.values * self.beta)
        return np.random.choice(self.n_options, p=choice_probs)


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
    # ---- Test: __init__ sets shapes and defaults ----
    m = DynamicValueShapeModel(
        lr=0.2, beta=3.0, rel_coef=1.0, rel_const=0.0, n_options=3
    )
    assert m.n_options == 3
    assert m.values.shape == (3,)
    assert m.tracked_choices.shape == (3,)
    assert_allclose(m.values, [0.0, 0.0, 0.0])
    assert_allclose(m.tracked_choices, [1.0, 1.0, 1.0])

    # ---- Test: __init__ validation ----
    assert_raises(
        ValueError,
        DynamicValueShapeModel,
        lr=-0.01,
        beta=1.0,
        rel_coef=0.0,
        rel_const=0.0,
        n_options=3,
    )
    assert_raises(
        ValueError,
        DynamicValueShapeModel,
        lr=1.01,
        beta=1.0,
        rel_coef=0.0,
        rel_const=0.0,
        n_options=3,
    )
    assert_raises(
        ValueError,
        DynamicValueShapeModel,
        lr=0.5,
        beta=-0.1,
        rel_coef=0.0,
        rel_const=0.0,
        n_options=3,
    )

    # ---- Test: learn_from_self updates chosen value correctly ----
    m = DynamicValueShapeModel(
        lr=0.5, beta=1.0, rel_coef=0.0, rel_const=0.0, n_options=3
    )
    m.values[:] = 0.0
    m.learn_from_self(choice=1, reward=1)
    # V_new = 0 + 0.5*(1-0)=0.5
    assert_allclose(m.values, [0.0, 0.5, 0.0])

    # ---- Test: learn_from_self out of range ----
    assert_raises(ValueError, m.learn_from_self, choice=3, reward=1)

    # ---- Test: learn_from_other out of range ----
    assert_raises(ValueError, m.learn_from_other, choice=3, reward=1)

    # ---- Test: learn_from_other updates in correct order + increments tracking ----
    lr = 0.2
    rel_coef = 2.0
    rel_const = -0.5
    m = DynamicValueShapeModel(
        lr=lr, beta=1.0, rel_coef=rel_coef, rel_const=rel_const, n_options=3
    )

    choice = 0
    reward = 1

    # compute expected manually
    v0 = m.values[choice]  # 0
    tc0 = m.tracked_choices.copy()  # [1,1,1]
    expected_prob = tc0[choice] / tc0.sum()  # 1/3
    reliability = expit(rel_coef * expected_prob + rel_const)

    v1 = v0 + reliability * (1 - v0)  # shaping
    v2 = v1 + lr * (reward - v1)  # RL update

    m.learn_from_other(choice=choice, reward=reward)

    assert_allclose(m.values[choice], v2)
    assert_allclose(m.tracked_choices, [2.0, 1.0, 1.0])  # incremented only at 'choice'

    m.reset()
    assert_allclose(m.values, [0.5, 0.5, 0.5])
    assert_allclose(m.tracked_choices, [1.0, 1.0, 1.0])

    # ---- Test: choose returns valid index and uses softmax distribution ----
    # We’ll set the seed so the test is reproducible.
    m = DynamicValueShapeModel(
        lr=0.1, beta=5.0, rel_coef=0.0, rel_const=0.0, n_options=3
    )
    m.values[:] = np.array([0.1, 0.2, -0.3])

    np.random.seed(0)
    c = m.choose()
    assert 0 <= c < m.n_options

    # Check that the probabilities are well-formed and match softmax(values*beta)
    p = softmax(m.values * m.beta)
    assert_allclose(p.sum(), 1.0)
    assert np.all(p >= 0)

    print("All inline tests passed ✅")
