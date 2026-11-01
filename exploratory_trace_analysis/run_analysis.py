"""Complete Stage 3 propagation, rejoining, and matched-state analysis pipeline.

Processes all 27 valid seeds across both targets (54 conditions):
1. Joint classification (Q, Action, Schedule, Return).
2. Trajectory persistence and absorbing rejoining analysis.
3. Matched-state decision analysis at first divergence.
4. Asymmetric case studies: Seed 3233 and Seed 9126.
5. Dream-phase Q-trajectory reconstruction across 60 updates.
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from exploratory_trace_analysis.trace_parser import (
    compute_q_l1,
    extract_actions,
    find_first_action_divergence,
    extract_collection_schedule,
    extract_return_reward,
    analyze_matched_state,
    analyze_rejoining_and_persistence,
    reconstruct_dream_q_trajectory,
    classify_profile
)

def run_full_analysis() -> Dict[str, Any]:
    manifest_path = Path("repeated_draws/manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError("repeated_draws/manifest.json not found")
        
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    valid_seeds = []
    for s in manifest["completed_seeds"]:
        p = Path(f"repeated_draws/seed_{s}.json")
        with open(p) as f:
            d = json.load(f)
        if d.get("status") == "VALID":
            valid_seeds.append((s, d))
            
    if len(valid_seeds) != 27:
        raise ValueError(f"Expected 27 valid seeds, found {len(valid_seeds)}")
        
    results = {
        "targets": {},
        "case_studies": {},
        "dream_q_divergence": []
    }
    
    for tgt in ["fischer_target", "mal_target"]:
        target_records = []
        for s, d in valid_seeds:
            arm_d = d["arms"][tgt]["dream"]
            arm_u = d["arms"][tgt]["dream_untransformed"]
            
            # 1. Metrics
            l1 = compute_q_l1(arm_d, arm_u)
            acts_d = extract_actions(arm_d)
            acts_u = extract_actions(arm_u)
            t_div = find_first_action_divergence(acts_d, acts_u)
            sched_d = extract_collection_schedule(arm_d["return_trace"])
            sched_u = extract_collection_schedule(arm_u["return_trace"])
            ret_d = extract_return_reward(arm_d)
            ret_u = extract_return_reward(arm_u)
            contrast = round(ret_d - ret_u, 4)
            
            q_div = l1 > 1e-4
            act_div = (t_div is not None)
            sched_div = (sched_d != sched_u)
            ret_div = (abs(contrast) > 1e-4)
            
            tup, profile_name = classify_profile(q_div, act_div, sched_div, ret_div)
            
            # 2. Rejoining and persistence
            rej = analyze_rejoining_and_persistence(arm_d["return_trace"], arm_u["return_trace"], t_div)
            
            # 3. Matched state analysis at t_div
            matched_info = None
            if t_div is not None:
                step_d = arm_d["return_trace"][t_div]
                step_u = arm_u["return_trace"][t_div]
                matched_info = analyze_matched_state(step_d, step_u)
                
            rec = {
                "seed": s,
                "q_l1": round(l1, 4),
                "q_div": q_div,
                "act_div": act_div,
                "t_div": t_div,
                "sched_div": sched_div,
                "ret_div": ret_div,
                "contrast": contrast,
                "profile_tuple": tup,
                "profile_name": profile_name,
                "persistence": rej["persistence"],
                "action_persistence": rej["action_persistence"],
                "rejoined": rej["rejoined"],
                "t_rejoin": rej["t_rejoin"],
                "transient_intersections_count": len(rej["transient_intersections"]),
                "matched_decision": matched_info
            }
            target_records.append(rec)
            
        results["targets"][tgt] = target_records

    # 4. Asymmetry Case Studies: Seed 3233 and Seed 9126
    for s_case in [3233, 9126]:
        d_case = next(d for s, d in valid_seeds if s == s_case)
        case_data = {
            "seed": s_case,
            "suspended_pos": d_case["suspended_pos"],
            "suspended_collected": d_case["suspended_collected"],
            "realized_dream": d_case["realized_dream"],
            "live_regime": d_case["live_regime_at_suspension"],
            "mal_swapped_pair": d_case.get("mal_swapped_pair"),
            "mal_swap_diff": d_case.get("mal_swap_diff"),
            "targets": {}
        }
        for tgt in ["fischer_target", "mal_target"]:
            arm_d = d_case["arms"][tgt]["dream"]
            arm_u = d_case["arms"][tgt]["dream_untransformed"]
            sched_d = extract_collection_schedule(arm_d["return_trace"])
            sched_u = extract_collection_schedule(arm_u["return_trace"])
            acts_d = extract_actions(arm_d)
            acts_u = extract_actions(arm_u)
            t_div = find_first_action_divergence(acts_d, acts_u)
            ret_d = extract_return_reward(arm_d)
            ret_u = extract_return_reward(arm_u)
            rej = analyze_rejoining_and_persistence(arm_d["return_trace"], arm_u["return_trace"], t_div)
            
            matched = None
            if t_div is not None:
                matched = analyze_matched_state(arm_d["return_trace"][t_div], arm_u["return_trace"][t_div])
                
            case_data["targets"][tgt] = {
                "q_l1": round(compute_q_l1(arm_d, arm_u), 4),
                "t_div": t_div,
                "sched_d": sched_d,
                "sched_u": sched_u,
                "ret_d": ret_d,
                "ret_u": ret_u,
                "contrast": round(ret_d - ret_u, 4),
                "matched": matched,
                "persistence": rej["persistence"],
                "rejoined": rej["rejoined"],
                "t_rejoin": rej["t_rejoin"]
            }
        results["case_studies"][str(s_case)] = case_data

    # 5. Dream Q Trajectory analysis
    for s, d in valid_seeds:
        for tgt in ["fischer_target", "mal_target"]:
            arm_d = d["arms"][tgt]["dream"]
            arm_u = d["arms"][tgt]["dream_untransformed"]
            traj_d = reconstruct_dream_q_trajectory(arm_d["q_table_before_dream"], arm_d["q_updates_dream"], arm_d["q_table_after_dream"])
            traj_u = reconstruct_dream_q_trajectory(arm_u["q_table_before_dream"], arm_u["q_updates_dream"], arm_u["q_table_after_dream"])
            
            step_div = None
            for step in range(61):
                qd = traj_d[step]
                qu = traj_u[step]
                all_k = set(qd.keys()) | set(qu.keys())
                l1 = sum(abs(qd.get(k, 0.0) - qu.get(k, 0.0)) for k in all_k)
                if l1 > 1e-4:
                    step_div = step
                    break
            results["dream_q_divergence"].append({
                "seed": s,
                "target": tgt,
                "first_q_div_step": step_div,
                "final_l1": round(compute_q_l1(arm_d, arm_u), 4)
            })

    return results

if __name__ == "__main__":
    res = run_full_analysis()
    out_path = Path("exploratory_trace_analysis/analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Analysis results saved to {out_path}")
