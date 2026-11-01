"""Independent trace parser for Remnant Stage 4/pilot runs.

Provides clean, standalone parsing and metric recomputation for:
- Post-dream Q-table divergence (L1)
- Action sequence divergence and first divergence step
- Collection schedule extraction and contrast
- Matched-state decision inspection at divergence point
- Trajectory persistence, transient state intersections, and absorbing rejoining
- Per-update dream phase Q trajectory reconstruction
"""
from typing import Dict, Any, List, Tuple, Optional
import math

def parse_q_table(q_table_list: List[Dict[str, Any]]) -> Dict[Tuple[Tuple[int, int], Tuple[int, ...], int], float]:
    """Converts a serialized Q-table list to a dict with hashable state-action keys."""
    table = {}
    for item in q_table_list:
        pos = tuple(item["pos"])
        col = tuple(sorted(item["col"]))
        act = int(item["action"])
        val = float(item["val"])
        table[(pos, col, act)] = val
    return table

def compute_q_l1(arm_a: Dict[str, Any], arm_b: Dict[str, Any], q_key: str = "q_table_after_dream") -> float:
    """Computes exact L1 distance between Q-tables of two arms."""
    qa = parse_q_table(arm_a.get(q_key, []))
    qb = parse_q_table(arm_b.get(q_key, []))
    all_keys = set(qa.keys()) | set(qb.keys())
    return sum(abs(qa.get(k, 0.0) - qb.get(k, 0.0)) for k in all_keys)

def extract_actions(arm: Dict[str, Any]) -> List[int]:
    """Extracts the 200 return actions."""
    if "return_actions" in arm and arm["return_actions"]:
        return [int(a) for a in arm["return_actions"]]
    return [int(step["chosen_action"]) for step in arm.get("return_trace", [])]

def find_first_action_divergence(actions_a: List[int], actions_b: List[int]) -> Optional[int]:
    """Returns the smallest step index t where actions differ, or None if identical."""
    min_len = min(len(actions_a), len(actions_b))
    for t in range(min_len):
        if actions_a[t] != actions_b[t]:
            return t
    if len(actions_a) != len(actions_b):
        return min_len
    return None

def extract_collection_schedule(return_trace: List[Dict[str, Any]]) -> List[Tuple[int, Tuple[int, ...]]]:
    """Extracts ordered list of (step, newly_collected_objects)."""
    sched = []
    for step in return_trace:
        cur_col = set(step["state"][1])
        next_col = set(step["next_state"][1])
        new_items = next_col - cur_col
        if new_items:
            sched.append((int(step["step"]), tuple(sorted(new_items))))
    return sched

def extract_return_reward(arm: Dict[str, Any]) -> float:
    """Extracts or computes the cumulative return reward."""
    if "return_cumreward" in arm and arm["return_cumreward"] is not None:
        return round(float(arm["return_cumreward"]), 4)
    total = sum(float(step.get("reward", 0.0)) for step in arm.get("return_trace", []))
    return round(total, 4)

def analyze_matched_state(step_d: Dict[str, Any], step_u: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes the agent decision at the matched divergence step."""
    is_rand_d = bool(step_d.get("is_random", False))
    is_rand_u = bool(step_u.get("is_random", False))
    
    if is_rand_d or is_rand_u:
        return {
            "category": "random_exploration_branch",
            "is_random_d": is_rand_d,
            "is_random_u": is_rand_u,
            "details": f"Exploration triggered (dream={is_rand_d}, untrans={is_rand_u})"
        }
    
    vals_d = step_d.get("blended_vals")
    vals_u = step_u.get("blended_vals")
    
    if vals_d is None or vals_u is None:
        return {
            "category": "insufficient_data",
            "is_random_d": is_rand_d,
            "is_random_u": is_rand_u,
            "details": "blended_vals missing from trace"
        }
    
    max_d = max(vals_d)
    max_u = max(vals_u)
    argmaxes_d = [i for i, v in enumerate(vals_d) if math.isclose(v, max_d, rel_tol=1e-7, abs_tol=1e-7)]
    argmaxes_u = [i for i, v in enumerate(vals_u) if math.isclose(v, max_u, rel_tol=1e-7, abs_tol=1e-7)]
    
    act_d = step_d["chosen_action"]
    act_u = step_u["chosen_action"]
    
    if len(argmaxes_d) > 1 or len(argmaxes_u) > 1:
        return {
            "category": "tie_breaking",
            "act_d": act_d,
            "act_u": act_u,
            "argmaxes_d": argmaxes_d,
            "argmaxes_u": argmaxes_u,
            "details": "Multiple actions tied for maximum value"
        }
        
    if argmaxes_d != argmaxes_u:
        return {
            "category": "greedy_value_shift",
            "act_d": act_d,
            "act_u": act_u,
            "argmax_d": argmaxes_d[0],
            "argmax_u": argmaxes_u[0],
            "vals_d": vals_d,
            "vals_u": vals_u,
            "details": f"Greedy argmax shifted: dream={argmaxes_d[0]}, untrans={argmaxes_u[0]}"
        }
        
    return {
        "category": "same_argmax_divergent_choice",
        "act_d": act_d,
        "act_u": act_u,
        "argmax_d": argmaxes_d[0],
        "argmax_u": argmaxes_u[0],
        "details": "Argmax matches but actions differ (possible unrecorded tie or state mismatch)"
    }

def analyze_rejoining_and_persistence(trace_d: List[Dict[str, Any]], trace_u: List[Dict[str, Any]],
                                      t_div: Optional[int]) -> Dict[str, Any]:
    """Computes state persistence, transient crossings, and absorbing rejoining."""
    if t_div is None:
        return {
            "diverged": False,
            "t_div": None,
            "persistence": 0.0,
            "action_persistence": 0.0,
            "transient_intersections": [],
            "rejoined": True,
            "t_rejoin": 0
        }
        
    horizon = min(len(trace_d), len(trace_u))
    post_div_steps = horizon - (t_div + 1)
    if post_div_steps <= 0:
        return {
            "diverged": True,
            "t_div": t_div,
            "persistence": 0.0,
            "action_persistence": 0.0,
            "transient_intersections": [],
            "rejoined": False,
            "t_rejoin": None
        }
        
    diff_state_count = 0
    diff_act_count = 0
    transient_intersections = []
    
    # State equality helper
    def states_match(s1, s2):
        return (tuple(s1[0]) == tuple(s2[0])) and (tuple(sorted(s1[1])) == tuple(sorted(s2[1])))
    
    for t in range(t_div + 1, horizon):
        s_d = trace_d[t]["state"]
        s_u = trace_u[t]["state"]
        a_d = trace_d[t]["chosen_action"]
        a_u = trace_u[t]["chosen_action"]
        
        s_match = states_match(s_d, s_u)
        a_match = (a_d == a_u)
        
        if not s_match:
            diff_state_count += 1
        else:
            transient_intersections.append(t)
            
        if not a_match:
            diff_act_count += 1
            
    persistence = diff_state_count / post_div_steps
    action_persistence = diff_act_count / post_div_steps
    
    # Check for absorbing rejoin: earliest step t_rej where for all t >= t_rej, state and action match
    rejoined = False
    t_rejoin = None
    for t in range(t_div + 1, horizon):
        tail_matches = True
        for k in range(t, horizon):
            if not states_match(trace_d[k]["state"], trace_u[k]["state"]) or (trace_d[k]["chosen_action"] != trace_u[k]["chosen_action"]):
                tail_matches = False
                break
        if tail_matches:
            rejoined = True
            t_rejoin = t
            break
            
    return {
        "diverged": True,
        "t_div": t_div,
        "persistence": round(persistence, 4),
        "action_persistence": round(action_persistence, 4),
        "transient_intersections": transient_intersections,
        "rejoined": rejoined,
        "t_rejoin": t_rejoin
    }

def reconstruct_dream_q_trajectory(q_before_list: List[Dict[str, Any]],
                                    q_updates: List[Dict[str, Any]],
                                    q_after_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[Tuple[Tuple[int, int], Tuple[int, ...], int], float]]:
    """Reconstructs full Q-table after each of the 60 dream updates.
    
    Returns list of 61 Q-tables: Q_0, Q_1, ..., Q_60.
    Optionally verifies that Q_60 matches q_after_list.
    """
    current_q = parse_q_table(q_before_list)
    trajectory = [dict(current_q)]
    
    for u in q_updates:
        pos = tuple(u["state"][0])
        col = tuple(sorted(u["state"][1]))
        act = int(u["action"])
        new_q = float(u["new_q"])
        current_q[(pos, col, act)] = new_q
        trajectory.append(dict(current_q))
        
    if q_after_list is not None:
        target_q = parse_q_table(q_after_list)
        all_keys = set(current_q.keys()) | set(target_q.keys())
        for k in all_keys:
            v_reconstructed = current_q.get(k, 0.0)
            v_target = target_q.get(k, 0.0)
            if abs(v_reconstructed - v_target) > 1e-4:
                raise ValueError(f"Q-table reconstruction mismatch at key {k}: {v_reconstructed} vs target {v_target}")
                
    return trajectory

def classify_profile(q_div: bool, act_div: bool, sched_div: bool, ret_div: bool) -> Tuple[Tuple[int, int, int, int], str]:
    """Classifies the observed 4-tuple of flags."""
    tup = (1 if q_div else 0, 1 if act_div else 0, 1 if sched_div else 0, 1 if ret_div else 0)
    
    if tup == (0, 0, 0, 0):
        name = "Exact Null (Seed-77 Pattern)"
    elif tup == (1, 0, 0, 0):
        name = "Latent Divergence (Q divergence only)"
    elif tup == (1, 1, 0, 0):
        name = "Behavioral Drift Without Schedule Impact (Seed-42 Pattern)"
    elif tup == (1, 1, 1, 0):
        name = "Schedule Divergence Without Return Impact"
    elif tup == (1, 1, 1, 1):
        name = "Complete Divergence (Q, Action, Schedule, Return)"
    elif tup == (1, 1, 0, 1):
        name = "Non-Nested: Return Changed Without Schedule Difference"
    elif tup[0] == 0 and any(tup[1:]):
        name = f"Non-Nested: Action/Schedule Divergence Without Q Divergence {tup}"
    else:
        name = f"Irregular Profile {tup}"
        
    return tup, name
