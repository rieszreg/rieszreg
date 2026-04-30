import pytest

from rieszboost.tracer import LinearForm, Tracer, trace


def test_ate_traces_to_two_terms():
    def m(alpha):
        def inner(z):
            return alpha(a=1, x=z["x"]) - alpha(a=0, x=z["x"])
        return inner

    pairs = trace(m, {"x": 0.5})
    assert len(pairs) == 2
    by_a = {pt["a"]: c for c, pt in pairs}
    assert by_a == {1: 1.0, 0: -1.0}
    assert all(pt["x"] == 0.5 for _, pt in pairs)


def test_scalar_multiplication_and_addition():
    def m(alpha):
        def inner(z):
            return 2.0 * alpha(a=1, x=z["x"]) + 0.5 * alpha(a=0, x=z["x"])
        return inner

    pairs = dict(((tuple(sorted(pt.items())), c) for c, pt in trace(m, {"x": 1.0})))
    assert pairs[(("a", 1), ("x", 1.0))] == 2.0
    assert pairs[(("a", 0), ("x", 1.0))] == 0.5


def test_duplicate_points_merge():
    def m(alpha):
        def inner(z):
            return alpha(a=1, x=z["x"]) + alpha(a=1, x=z["x"])
        return inner

    pairs = trace(m, {"x": 0.0})
    assert len(pairs) == 1
    assert pairs[0][0] == 2.0


def test_zero_coef_dropped():
    def m(alpha):
        def inner(z):
            return alpha(a=1, x=z["x"]) - alpha(a=1, x=z["x"])
        return inner

    assert trace(m, {"x": 0.0}) == []


def test_nonlinear_op_raises():
    def m(alpha):
        def inner(z):
            return alpha(a=1, x=z["x"]) ** 2
        return inner

    with pytest.raises(TypeError):
        trace(m, {"x": 0.0})


def test_constant_offset_raises():
    def m(alpha):
        def inner(z):
            return alpha(a=1, x=z["x"]) + 1.0
        return inner

    with pytest.raises(TypeError):
        trace(m, {"x": 0.0})


def test_alpha_times_alpha_raises():
    def m(alpha):
        def inner(z):
            return alpha(a=1, x=z["x"]) * alpha(a=0, x=z["x"])
        return inner

    with pytest.raises(TypeError):
        trace(m, {"x": 0.0})


def test_shift_estimand_traces():
    def m(alpha):
        def inner(z):
            return alpha(a=z["a"] + 1.0, x=z["x"]) - alpha(a=z["a"], x=z["x"])
        return inner

    pairs = trace(m, {"a": 0.3, "x": 1.0})
    coefs_by_a = {pt["a"]: c for c, pt in pairs}
    assert coefs_by_a == {1.3: 1.0, 0.3: -1.0}
