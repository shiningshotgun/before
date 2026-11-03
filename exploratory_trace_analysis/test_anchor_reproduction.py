"""Anchor reproduction tests verifying known pilot values using the independent trace parser."""
import json
from pathlib import Path
import pytest
from exploratory_trace_analysis import pilot_data_present
from exploratory_trace_analysis.trace_parser import (
    compute_q_l1,
    extract_actions,
    find_first_action_divergence,
    extract_collection_schedule,
    extract_return_reward,
    classify_profile
)

pytestmark = pytest.mark.skipif(
    not pilot_data_present(),
    reason="repeated_draws/ raw traces not present (release v1.0-pilot-data)"
)

def get_valid_seed_traces():
    manifest_path = Path("repeated_draws/manifest.json")
    assert manifest_path.exists(), "Manifest file not found"
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    valid_traces = []
    for s in manifest["completed_seeds"]:
        p = Path(f"repeated_draws/seed_{s}.json")
        with open(p) as f:
            d = json.load(f)
        if d.get("status") == "VALID":
            valid_traces.append((s, d))
    return valid_traces

def test_anchor_reproduction():
    valid_traces = get_valid_seed_traces()
    assert len(valid_traces) == 27, f"Expected 27 valid seeds, found {len(valid_traces)}"
    
    results = {}
    for tgt in ["fischer_target", "mal_target"]:
        q_l1_pos = []
        act_divs = []
        div_steps = []
        sched_diffs = []
        contrasts = []
        contrasts_by_seed = {}
        archetypes = {"null": 0, "pattern": 0, "divergent": 0}
        
        for s, d in valid_traces:
            arm_d = d["arms"][tgt]["dream"]
            arm_u = d["arms"][tgt]["dream_untransformed"]
            
            # Post-dream Q L1
            l1 = compute_q_l1(arm_d, arm_u)
            if l1 > 1e-4:
                q_l1_pos.append(s)
                
            # Actions
            acts_d = extract_actions(arm_d)
            acts_u = extract_actions(arm_u)
            t_div = find_first_action_divergence(acts_d, acts_u)
            if t_div is not None:
                act_divs.append(s)
                div_steps.append(t_div)
                
            # Schedule
            sched_d = extract_collection_schedule(arm_d["return_trace"])
            sched_u = extract_collection_schedule(arm_u["return_trace"])
            if sched_d != sched_u:
                sched_diffs.append(s)
                
            # Contrast
            ret_d = extract_return_reward(arm_d)
            ret_u = extract_return_reward(arm_u)
            c = round(ret_d - ret_u, 3)
            contrasts.append(c)
            contrasts_by_seed[s] = c
            
            # Archetypes
            if sched_d != sched_u:
                archetypes["divergent"] += 1
            elif l1 > 1e-4:
                archetypes["pattern"] += 1
            else:
                archetypes["null"] += 1
                
        results[tgt] = {
            "q_l1_count": len(q_l1_pos),
            "q_l1_seeds": q_l1_pos,
            "act_div_count": len(act_divs),
            "act_div_seeds": act_divs,
            "mean_div_step": round(sum(div_steps) / len(div_steps), 1),
            "sched_diff_count": len(sched_diffs),
            "sched_diff_seeds": sched_diffs,
            "archetypes": archetypes,
            "contrasts": contrasts,
            "contrasts_by_seed": contrasts_by_seed
        }
        
    f_res = results["fischer_target"]
    m_res = results["mal_target"]
    
    # 1. Post-dream Q L1 > 0 count: exactly 14 per target
    assert f_res["q_l1_count"] == 14, f"Fischer Q L1 count {f_res['q_l1_count']} != 14"
    assert m_res["q_l1_count"] == 14, f"Mal Q L1 count {m_res['q_l1_count']} != 14"
    
    # 2. Action divergence counts: 12 Fischer, 11 Mal
    assert f_res["act_div_count"] == 12, f"Fischer act div count {f_res['act_div_count']} != 12"
    assert m_res["act_div_count"] == 11, f"Mal act div count {m_res['act_div_count']} != 11"
    
    # 3. First action divergence step mean: 71.8 Fischer, 63.4 Mal
    assert f_res["mean_div_step"] == 71.8, f"Fischer mean step {f_res['mean_div_step']} != 71.8"
    assert m_res["mean_div_step"] == 63.4, f"Mal mean step {m_res['mean_div_step']} != 63.4"
    
    # 4. Archetype partition: exactly 13 null, 6 pattern, 8 divergent
    assert f_res["archetypes"] == {"null": 13, "pattern": 6, "divergent": 8}
    assert m_res["archetypes"] == {"null": 13, "pattern": 6, "divergent": 8}
    
    # 5. Schedule differences: 8/27 per target with 7 shared seeds
    assert f_res["sched_diff_count"] == 8
    assert m_res["sched_diff_count"] == 8
    shared_sched = set(f_res["sched_diff_seeds"]) & set(m_res["sched_diff_seeds"])
    assert len(shared_sched) == 7
    assert sorted(shared_sched) == [1978, 2692, 3327, 4054, 4915, 6406, 7356]
    
    # 6. Specific seed properties:
    # Seeds 1022 and 7763 have Q divergence with zero action divergence in both targets
    assert 1022 in f_res["q_l1_seeds"] and 1022 not in f_res["act_div_seeds"]
    assert 7763 in f_res["q_l1_seeds"] and 7763 not in f_res["act_div_seeds"]
    assert 1022 in m_res["q_l1_seeds"] and 1022 not in m_res["act_div_seeds"]
    assert 7763 in m_res["q_l1_seeds"] and 7763 not in m_res["act_div_seeds"]
    
    # Seed 3233: Q divergence with zero action divergence in Mal only
    assert 3233 in m_res["q_l1_seeds"] and 3233 not in m_res["act_div_seeds"]
    # Seed 3233 changes schedule for Fischer only
    assert 3233 in f_res["sched_diff_seeds"] and 3233 not in m_res["sched_diff_seeds"]
    # Seed 9126 changes schedule for Mal only
    assert 9126 in m_res["sched_diff_seeds"] and 9126 not in f_res["sched_diff_seeds"]

    # 7. Return contrast distribution assertions
    # Fischer: 25 zero, 0 positive, 2 negative (4054 = -1.0, 4915 = -0.5)
    f_c = f_res["contrasts"]
    assert sum(1 for c in f_c if c == 0.0) == 25, f"Fischer zero contrasts count {sum(1 for c in f_c if c == 0.0)} != 25"
    assert sum(1 for c in f_c if c > 0.0) == 0, f"Fischer positive contrasts count {sum(1 for c in f_c if c > 0.0)} != 0"
    assert sum(1 for c in f_c if c < 0.0) == 2, f"Fischer negative contrasts count {sum(1 for c in f_c if c < 0.0)} != 2"
    assert f_res["contrasts_by_seed"][4054] == -1.0, f"Fischer seed 4054 contrast {f_res['contrasts_by_seed'][4054]} != -1.0"
    assert f_res["contrasts_by_seed"][4915] == -0.5, f"Fischer seed 4915 contrast {f_res['contrasts_by_seed'][4915]} != -0.5"

    # Mal: 23 zero, 2 positive (2692 = +1.5, 4054 = +1.0), 2 negative (3327 = -0.5, 9126 = -1.0)
    m_c = m_res["contrasts"]
    assert sum(1 for c in m_c if c == 0.0) == 23, f"Mal zero contrasts count {sum(1 for c in m_c if c == 0.0)} != 23"
    assert sum(1 for c in m_c if c > 0.0) == 2, f"Mal positive contrasts count {sum(1 for c in m_c if c > 0.0)} != 2"
    assert sum(1 for c in m_c if c < 0.0) == 2, f"Mal negative contrasts count {sum(1 for c in m_c if c < 0.0)} != 2"
    assert m_res["contrasts_by_seed"][2692] == 1.5, f"Mal seed 2692 contrast {m_res['contrasts_by_seed'][2692]} != +1.5"
    assert m_res["contrasts_by_seed"][4054] == 1.0, f"Mal seed 4054 contrast {m_res['contrasts_by_seed'][4054]} != +1.0"
    assert m_res["contrasts_by_seed"][3327] == -0.5, f"Mal seed 3327 contrast {m_res['contrasts_by_seed'][3327]} != -0.5"
    assert m_res["contrasts_by_seed"][9126] == -1.0, f"Mal seed 9126 contrast {m_res['contrasts_by_seed'][9126]} != -1.0"
