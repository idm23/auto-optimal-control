"""Certificates: properties of a problem's expressions, proved exactly or by construction."""

import warnings

import sympy as sp

from auto_optimal_control import examples
from auto_optimal_control.certificates import certify
from auto_optimal_control.policy import SecondaryCost, analyze
from auto_optimal_control.problem import Box, OptimalControlProblem, PathConstraint


def with_jerk_cost(problem):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SecondaryCost().apply(problem, analyze(problem))


def test_a_jerk_limited_axis_with_a_one_sided_speed_limit_is_not_time_reversible():
    c = certify(with_jerk_cost(examples.jerk_limited_move()))
    assert c["control_affine"] and c["strictly_convex_in_control"]
    assert c["cyclic_states"].detail == [sp.Symbol("x", real=True)]
    # Reversal flips v; the problem limits v <= V but not -v <= V.
    assert c["time_reversible"].holds is False
    assert c["time_reversible"].detail["state_signs"] == [1, -1, 1]


def test_two_sided_limits_make_the_axis_time_reversible():
    t, x, v, a, j = sp.symbols("t x v a j", real=True)
    V = sp.Symbol("V", positive=True)
    p = OptimalControlProblem(
        time=t, states=[x, v, a], controls=[j], dynamics=[v, a, j], running_cost=1 + j ** 2,
        initial_state=[0, 0, 0], final_state={x: 1, v: 0, a: 0}, params=[V],
        control_sets=[Box(j, -1, 1)],
        path_constraints=[PathConstraint(v - V, "v_max"), PathConstraint(-v - V, "v_min")])
    c = certify(p)
    assert c["time_reversible"].holds is True and c["time_reversible"].how == "exact"
    assert c["time_reversible"].detail["control_signs"] == [-1]


def test_the_two_link_arm_is_time_reversible_and_its_shoulder_angle_is_cyclic():
    c = certify(with_jerk_cost(examples.two_link_arm_min_time()))
    assert c["time_reversible"].holds is True and c["time_reversible"].how == "exact"
    assert [s.name for s in c["cyclic_states"].detail] == ["q1"]


def test_the_six_joint_arm_certifies_its_mass_matrix_and_reversal_by_construction():
    c = certify(with_jerk_cost(examples.six_joint_arm_min_time()))
    assert c["mass_matrix_positive_definite"].holds is True
    assert c["mass_matrix_positive_definite"].how == "by construction"
    assert c["time_reversible"].holds is True and c["time_reversible"].how == "by construction"
    rows = c["limit_control_rows"].detail
    assert rows["tau3_min"] == (2, -1) and rows["tau6_max"] == (5, 1)
    # Gravity acts along the base joint's axis, so the base angle appears nowhere.
    assert [s.name for s in c["cyclic_states"].detail] == ["q1"]
