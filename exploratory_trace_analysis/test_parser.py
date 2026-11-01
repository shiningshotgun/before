"""Tests for the independent trace parser using the Stage 1 synthetic fixtures."""
import pytest
from exploratory_trace_analysis.synthetic_fixtures import (
    make_fixture_null,
    make_fixture_divergent_persistent,
    make_fixture_rejoining,
    make_fixture_non_nested_latent
)
from exploratory_trace_analysis.trace_parser import (
    compute_q_l1,
    extract_actions,
    find_first_action_divergence,
    extract_collection_schedule,
    extract_return_reward,
    analyze_matched_state,
    analyze_rejoining_and_persistence,
    classify_profile
)

def test_fixture_null():
    fix = make_fixture_null()
    arm_d = fix["arms"]["fischer_target"]["dream"]
    arm_u = fix["arms"]["fischer_target"]["dream_untransformed"]
    exp = fix["expected"]
    
    q_l1 = compute_q_l1(arm_d, arm_u)
    assert (q_l1 > 0) == bool(exp["q_div"])
    
    acts_d = extract_actions(arm_d)
    acts_u = extract_actions(arm_u)
    t_div = find_first_action_divergence(acts_d, acts_u)
    assert t_div == exp["t_div"]
    
    sched_d = extract_collection_schedule(arm_d["return_trace"])
    sched_u = extract_collection_schedule(arm_u["return_trace"])
    assert (sched_d != sched_u) == bool(exp["sched_div"])
    
    ret_d = extract_return_reward(arm_d)
    ret_u = extract_return_reward(arm_u)
    assert (abs(ret_d - ret_u) > 1e-4) == bool(exp["ret_div"])
    
    rej = analyze_rejoining_and_persistence(arm_d["return_trace"], arm_u["return_trace"], t_div)
    assert rej["rejoined"] == exp["rejoined"]
    assert rej["persistence"] == exp["persistence"]
    
    tup, name = classify_profile(q_l1 > 0, t_div is not None, sched_d != sched_u, abs(ret_d - ret_u) > 1e-4)
    assert tup == (exp["q_div"], exp["act_div"], exp["sched_div"], exp["ret_div"])
    assert "Exact Null" in name

def test_fixture_divergent_persistent():
    fix = make_fixture_divergent_persistent()
    arm_d = fix["arms"]["fischer_target"]["dream"]
    arm_u = fix["arms"]["fischer_target"]["dream_untransformed"]
    exp = fix["expected"]
    
    q_l1 = compute_q_l1(arm_d, arm_u)
    assert (q_l1 > 0) == bool(exp["q_div"])
    
    acts_d = extract_actions(arm_d)
    acts_u = extract_actions(arm_u)
    t_div = find_first_action_divergence(acts_d, acts_u)
    assert t_div == exp["t_div"]
    
    sched_d = extract_collection_schedule(arm_d["return_trace"])
    sched_u = extract_collection_schedule(arm_u["return_trace"])
    assert (sched_d != sched_u) == bool(exp["sched_div"])
    
    ret_d = extract_return_reward(arm_d)
    ret_u = extract_return_reward(arm_u)
    assert (abs(ret_d - ret_u) > 1e-4) == bool(exp["ret_div"])
    
    matched = analyze_matched_state(arm_d["return_trace"][t_div], arm_u["return_trace"][t_div])
    assert matched["category"] == "greedy_value_shift"
    
    rej = analyze_rejoining_and_persistence(arm_d["return_trace"], arm_u["return_trace"], t_div)
    assert rej["rejoined"] == exp["rejoined"]
    assert rej["persistence"] == exp["persistence"]
    assert rej["t_rejoin"] == exp["t_rejoin"]
    
    tup, name = classify_profile(q_l1 > 0, t_div is not None, sched_d != sched_u, abs(ret_d - ret_u) > 1e-4)
    assert tup == (exp["q_div"], exp["act_div"], exp["sched_div"], exp["ret_div"])
    assert "Complete Divergence" in name

def test_fixture_rejoining():
    fix = make_fixture_rejoining()
    arm_d = fix["arms"]["fischer_target"]["dream"]
    arm_u = fix["arms"]["fischer_target"]["dream_untransformed"]
    exp = fix["expected"]
    
    q_l1 = compute_q_l1(arm_d, arm_u)
    assert (q_l1 > 0) == bool(exp["q_div"])
    
    acts_d = extract_actions(arm_d)
    acts_u = extract_actions(arm_u)
    t_div = find_first_action_divergence(acts_d, acts_u)
    assert t_div == exp["t_div"]
    
    sched_d = extract_collection_schedule(arm_d["return_trace"])
    sched_u = extract_collection_schedule(arm_u["return_trace"])
    assert (sched_d != sched_u) == bool(exp["sched_div"])
    
    ret_d = extract_return_reward(arm_d)
    ret_u = extract_return_reward(arm_u)
    assert (abs(ret_d - ret_u) > 1e-4) == bool(exp["ret_div"])
    
    rej = analyze_rejoining_and_persistence(arm_d["return_trace"], arm_u["return_trace"], t_div)
    assert rej["rejoined"] == exp["rejoined"]
    assert rej["t_rejoin"] == exp["t_rejoin"]
    assert rej["persistence"] < 1.0

def test_fixture_non_nested_latent():
    fix = make_fixture_non_nested_latent()
    arm_d = fix["arms"]["fischer_target"]["dream"]
    arm_u = fix["arms"]["fischer_target"]["dream_untransformed"]
    exp = fix["expected"]
    
    q_l1 = compute_q_l1(arm_d, arm_u)
    assert (q_l1 > 0) == bool(exp["q_div"])
    
    acts_d = extract_actions(arm_d)
    acts_u = extract_actions(arm_u)
    t_div = find_first_action_divergence(acts_d, acts_u)
    assert t_div is None
    
    sched_d = extract_collection_schedule(arm_d["return_trace"])
    sched_u = extract_collection_schedule(arm_u["return_trace"])
    assert (sched_d != sched_u) == bool(exp["sched_div"])
    
    ret_d = extract_return_reward(arm_d)
    ret_u = extract_return_reward(arm_u)
    assert (abs(ret_d - ret_u) > 1e-4) == bool(exp["ret_div"])
    
    tup, name = classify_profile(q_l1 > 0, t_div is not None, sched_d != sched_u, abs(ret_d - ret_u) > 1e-4)
    assert tup == (exp["q_div"], exp["act_div"], exp["sched_div"], exp["ret_div"])
    assert "Latent Divergence" in name
