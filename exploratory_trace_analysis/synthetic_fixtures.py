"""Synthetic trace fixtures for testing the independent trace parser.

Defines four synthetic fixtures matching the specifications in PLAN.md:
1. fixture_null: Identical Q, identical 200-step return, identical collections and return.
2. fixture_divergent_persistent: Diverges at step 20, stays separated, different collections and return.
3. fixture_rejoining: Diverges at step 10, absorbs/rejoins at step 30, identical collections and return.
4. fixture_non_nested_latent: Post-dream Q differs (L1 > 0), but return actions/states never diverge.
"""
from typing import Dict, Any, List

def build_q_table_list(q_dict: Dict[tuple, float]) -> List[Dict[str, Any]]:
    """Converts {(pos, col, action): val} to serialized list format."""
    out = []
    for (pos, col, act), val in q_dict.items():
        out.append({
            "pos": list(pos),
            "col": list(col),
            "action": act,
            "val": float(val)
        })
    return out

def build_return_step(step: int, pos: List[int], col: List[int], action: int,
                      is_random: bool = False, q_val: float = 1.0, reward: float = -0.01,
                      next_pos: List[int] = None, next_col: List[int] = None) -> Dict[str, Any]:
    """Constructs a single step dict for return_trace."""
    if next_pos is None:
        next_pos = list(pos)
    if next_col is None:
        next_col = list(col)
    
    q_raws = {str(a): (q_val if a == action else q_val - 0.5) for a in range(4)} if not is_random else None
    blended = [(q_val if a == action else q_val - 0.5) for a in range(4)] if not is_random else None
    
    return {
        "step": step,
        "state": [list(pos), list(col)],
        "chosen_action": action,
        "is_random": is_random,
        "q_raws": q_raws,
        "m_vals": {str(a): 0.0 for a in range(4)} if not is_random else None,
        "blended_vals": blended,
        "memory_lookups": [],
        "reward": reward,
        "next_state": [next_pos, next_col]
    }

def make_fixture_null() -> Dict[str, Any]:
    """1. Null Archetype: Q L1 = 0, exact action match, exact schedule match."""
    q_base = {((0, 0), (), 0): 0.5, ((0, 0), (), 1): 0.2}
    q_list = build_q_table_list(q_base)
    
    steps = [build_return_step(t, [0, 0], [], 0, is_random=False, q_val=0.5, reward=-0.01) for t in range(200)]
    
    arm_data = {
        "return_cumreward": -2.0,
        "q_table_before_dream": q_list,
        "q_table_after_dream": q_list,
        "q_updates_dream": [],
        "return_trace": steps,
        "return_actions": [0] * 200
    }
    
    return {
        "seed": 9001,
        "status": "VALID",
        "arms": {
            "fischer_target": {
                "dream": arm_data,
                "dream_untransformed": arm_data,
                "no_dream": arm_data
            }
        },
        "expected": {
            "q_div": 0,
            "act_div": 0,
            "sched_div": 0,
            "ret_div": 0,
            "t_div": None,
            "persistence": 0.0,
            "rejoined": True,
            "t_rejoin": 0
        }
    }

def make_fixture_divergent_persistent() -> Dict[str, Any]:
    """2. Persistent Divergent Archetype: Q L1 > 0, diverges at t=20, stays separated through t=199."""
    q_d = build_q_table_list({((2, 2), (), 0): 1.5, ((2, 2), (), 1): 0.5})
    q_u = build_q_table_list({((2, 2), (), 0): 0.5, ((2, 2), (), 1): 1.5})
    
    steps_d = []
    steps_u = []
    for t in range(200):
        if t < 20:
            step = build_return_step(t, [t // 10, t % 10], [], 0, is_random=False)
            steps_d.append(step)
            steps_u.append(step)
        elif t == 20:
            # First divergence step at matched state [2, 2]
            steps_d.append(build_return_step(20, [2, 2], [], 0, is_random=False, q_val=1.5, next_pos=[2, 3]))
            steps_u.append(build_return_step(20, [2, 2], [], 1, is_random=False, q_val=1.5, next_pos=[3, 2]))
        else:
            # Distinct coordinates and different collected items
            col_d_cur = [1] if t > 50 else []
            col_d_next = [1] if t >= 50 else []
            col_u_cur = [2] if t > 80 else []
            col_u_next = [2] if t >= 80 else []
            steps_d.append(build_return_step(t, [2, (t % 8)], col_d_cur, 0, reward=1.0 if t == 50 else -0.01, next_col=col_d_next))
            steps_u.append(build_return_step(t, [5, (t % 8)], col_u_cur, 1, reward=1.0 if t == 80 else -0.01, next_col=col_u_next))
            
    return {
        "seed": 9002,
        "status": "VALID",
        "arms": {
            "fischer_target": {
                "dream": {
                    "return_cumreward": 10.5,
                    "q_table_before_dream": q_u,
                    "q_table_after_dream": q_d,
                    "q_updates_dream": [],
                    "return_trace": steps_d,
                    "return_actions": [s["chosen_action"] for s in steps_d]
                },
                "dream_untransformed": {
                    "return_cumreward": 5.0,
                    "q_table_before_dream": q_u,
                    "q_table_after_dream": q_u,
                    "q_updates_dream": [],
                    "return_trace": steps_u,
                    "return_actions": [s["chosen_action"] for s in steps_u]
                }
            }
        },
        "expected": {
            "q_div": 1,
            "act_div": 1,
            "sched_div": 1,
            "ret_div": 1,
            "t_div": 20,
            "persistence": 1.0,
            "rejoined": False,
            "t_rejoin": None
        }
    }

def make_fixture_rejoining() -> Dict[str, Any]:
    """3. Absorbing Rejoin Archetype: diverges at t=10, absorbs/rejoins at t=30 through t=199."""
    q_d = build_q_table_list({((1, 1), (), 0): 1.5, ((1, 1), (), 1): 0.5})
    q_u = build_q_table_list({((1, 1), (), 0): 0.5, ((1, 1), (), 1): 1.5})
    
    steps_d = []
    steps_u = []
    for t in range(200):
        if t < 10:
            step = build_return_step(t, [1, 0], [], 0, is_random=False)
            steps_d.append(step)
            steps_u.append(step)
        elif t == 10:
            # Divergence at matched state [1, 1]
            steps_d.append(build_return_step(10, [1, 1], [], 0, next_pos=[1, 2]))
            steps_u.append(build_return_step(10, [1, 1], [], 1, next_pos=[2, 1]))
        elif 10 < t < 30:
            # Different paths
            steps_d.append(build_return_step(t, [1, (t % 5)], [], 0, next_pos=[1, (t % 5) + 1]))
            steps_u.append(build_return_step(t, [(t % 5), 1], [], 1, next_pos=[(t % 5) + 1, 1]))
        else:
            # Rejoin at state [3, 3] from step 30 through 199 with identical actions
            step = build_return_step(t, [3, 3], [], 2, is_random=False, next_pos=[3, 4])
            steps_d.append(step)
            steps_u.append(step)
            
    return {
        "seed": 9003,
        "status": "VALID",
        "arms": {
            "fischer_target": {
                "dream": {
                    "return_cumreward": -2.0,
                    "q_table_before_dream": q_u,
                    "q_table_after_dream": q_d,
                    "q_updates_dream": [],
                    "return_trace": steps_d,
                    "return_actions": [s["chosen_action"] for s in steps_d]
                },
                "dream_untransformed": {
                    "return_cumreward": -2.0,
                    "q_table_before_dream": q_u,
                    "q_table_after_dream": q_u,
                    "q_updates_dream": [],
                    "return_trace": steps_u,
                    "return_actions": [s["chosen_action"] for s in steps_u]
                }
            }
        },
        "expected": {
            "q_div": 1,
            "act_div": 1,
            "sched_div": 0,
            "ret_div": 0,
            "t_div": 10,
            "rejoined": True,
            "t_rejoin": 30
        }
    }

def make_fixture_non_nested_latent() -> Dict[str, Any]:
    """4. Non-Nested Latent Divergence Archetype: Q L1 > 0 on unvisited state, zero return action divergence."""
    # Modified state [9, 9] is never visited in return phase
    q_d = build_q_table_list({((0, 0), (), 0): 0.5, ((9, 9), (), 0): 2.0})
    q_u = build_q_table_list({((0, 0), (), 0): 0.5, ((9, 9), (), 0): 0.0})
    
    steps = [build_return_step(t, [0, 0], [], 0, is_random=False, q_val=0.5, reward=-0.01) for t in range(200)]
    
    return {
        "seed": 9004,
        "status": "VALID",
        "arms": {
            "fischer_target": {
                "dream": {
                    "return_cumreward": -2.0,
                    "q_table_before_dream": q_u,
                    "q_table_after_dream": q_d,
                    "q_updates_dream": [],
                    "return_trace": steps,
                    "return_actions": [0] * 200
                },
                "dream_untransformed": {
                    "return_cumreward": -2.0,
                    "q_table_before_dream": q_u,
                    "q_table_after_dream": q_u,
                    "q_updates_dream": [],
                    "return_trace": steps,
                    "return_actions": [0] * 200
                }
            }
        },
        "expected": {
            "q_div": 1,
            "act_div": 0,
            "sched_div": 0,
            "ret_div": 0,
            "t_div": None,
            "rejoined": True,
            "t_rejoin": 0
        }
    }
