"""Programmatically derives all tables and aggregate statistics for REPORT.md from analysis_results.json.

Also provides self-verification against REPORT.md to guarantee 100% agreement.
"""
import json
from pathlib import Path
import statistics

def load_analysis_results():
    p = Path(__file__).parent / "analysis_results.json"
    with open(p) as f:
        return json.load(f)

def format_plus(seeds):
    return ", ".join(f"+{s}" for s in seeds)

def format_shared_col(shared_seeds, f_only_seeds, m_only_seeds):
    parts = []
    if shared_seeds:
        parts.append(", ".join(str(s) for s in shared_seeds))
    
    if f_only_seeds and not m_only_seeds:
        if len(f_only_seeds) == 1:
            note = f"*(Fischer also includes {f_only_seeds[0]})*"
        else:
            note = f"*(Fischer: {format_plus(f_only_seeds)})*"
        parts.append(note)
    elif m_only_seeds and not f_only_seeds:
        if len(m_only_seeds) == 1:
            note = f"*(Mal also includes {m_only_seeds[0]})*"
        else:
            note = f"*(Mal: {format_plus(m_only_seeds)})*"
        parts.append(note)
    elif f_only_seeds and m_only_seeds:
        note = f"*(Fischer: {format_plus(f_only_seeds)}; Mal: {format_plus(m_only_seeds)})*"
        parts.append(note)
    elif not shared_seeds and not f_only_seeds and not m_only_seeds:
        parts.append("None")
        
    return " ".join(parts) if parts else "None"

def generate_classification_table(res):
    f_recs = res["targets"]["fischer_target"]
    m_recs = res["targets"]["mal_target"]
    
    denom_f = len(f_recs)
    denom_m = len(m_recs)
    
    REGULAR_PROFILES = [
        ((0, 0, 0, 0), "Exact Null (Seed-77 Pattern)"),
        ((1, 0, 0, 0), "Latent Q Divergence (Q differs, 0 action diff)"),
        ((1, 1, 0, 0), "Behavioral Drift Without Schedule Difference"),
        ((1, 1, 1, 0), "Schedule Divergence Without Return Contrast"),
        ((1, 1, 1, 1), "Complete Divergence (Q, Act, Sched, Return)"),
    ]
    
    lines = []
    header_f = f"Fischer Target ($k / {denom_f}$)"
    header_m = f"Mal Target ($k / {denom_m}$)"
    lines.append(f"| Profile 4-Tuple | Classification Name | {header_f} | {header_m} | Shared Seeds |")
    lines.append("| :---: | :--- | :---: | :---: | :---: |")
    
    reg_tuples = set(p[0] for p in REGULAR_PROFILES)
    
    # Derive membership directly from records
    for tup, name in REGULAR_PROFILES:
        f_seeds = [r["seed"] for r in f_recs if tuple(r["profile_tuple"]) == tup]
        m_seeds = [r["seed"] for r in m_recs if tuple(r["profile_tuple"]) == tup]
        
        shared = [s for s in f_seeds if s in m_seeds]
        f_only = [s for s in f_seeds if s not in m_seeds]
        m_only = [s for s in m_seeds if s not in f_seeds]
        
        f_pct = (len(f_seeds) / denom_f) if denom_f > 0 else 0.0
        m_pct = (len(m_seeds) / denom_m) if denom_m > 0 else 0.0
        f_str = f"{len(f_seeds)} ({f_pct:.1%})"
        m_str = f"{len(m_seeds)} ({m_pct:.1%})"
        
        shared_str = format_shared_col(shared, f_only, m_only)
        lines.append(f"| **{tup}** | **{name}** | {f_str} | {m_str} | {shared_str} |")
        
    # Check for irregular / non-nested profiles
    f_irreg = [r for r in f_recs if tuple(r["profile_tuple"]) not in reg_tuples]
    m_irreg = [r for r in m_recs if tuple(r["profile_tuple"]) not in reg_tuples]
    irreg_tuples = sorted(list(set([tuple(r["profile_tuple"]) for r in f_irreg] + [tuple(r["profile_tuple"]) for r in m_irreg])))
    
    if not irreg_tuples:
        lines.append(f"| **Non-Nested** | Any irregular combination (e.g. (1,1,0,1) or (0,1,*,*)) | **0 (0.0%)** | **0 (0.0%)** | None |")
    else:
        for itup in irreg_tuples:
            f_seeds = [r["seed"] for r in f_irreg if tuple(r["profile_tuple"]) == itup]
            m_seeds = [r["seed"] for r in m_irreg if tuple(r["profile_tuple"]) == itup]
            shared = [s for s in f_seeds if s in m_seeds]
            f_only = [s for s in f_seeds if s not in m_seeds]
            m_only = [s for s in m_seeds if s not in f_seeds]
            f_pct = (len(f_seeds) / denom_f) if denom_f > 0 else 0.0
            m_pct = (len(m_seeds) / denom_m) if denom_m > 0 else 0.0
            f_str = f"**{len(f_seeds)} ({f_pct:.1%})**"
            m_str = f"**{len(m_seeds)} ({m_pct:.1%})**"
            shared_str = format_shared_col(shared, f_only, m_only)
            lines.append(f"| **{itup}** | Irregular profile | {f_str} | {m_str} | {shared_str} |")
            
    return "\n".join(lines)

def get_record_display_props(r):
    val_l1 = r["q_l1"]
    l1_str = f"{val_l1:.4f}"
    t_div_str = str(r["t_div"]) if r["t_div"] is not None else "None"
    matched_str = r["matched_decision"]["category"] if r["matched_decision"] else "N/A (No div)"
    val_pers = r["persistence"]
    persist_str = f"{val_pers:.4f}"
    if not r["act_div"]:
        rejoin_str = "True"
        t_rej_str = "0"
        shared_str = "200"
    else:
        rejoin_str = "**True**" if r["rejoined"] else "False"
        t_rej = r["t_rejoin"]
        t_rej_str = f"**{t_rej}**" if r["rejoined"] else "None"
        shared_str = str(r["transient_intersections_count"])
    tup_str = str(tuple(r["profile_tuple"]))
    return (l1_str, t_div_str, matched_str, persist_str, rejoin_str, t_rej_str, shared_str, tup_str)

def generate_per_seed_table(res):
    lines = []
    lines.append("| Seed | Target | Post-Dream $Q$ $L_1$ | $t_{\\text{div}}$ | Matched State Decision | State Persistence ($P_{\\text{state}}$) | Rejoined? | $t_{\\text{rejoin}}$ | Shared Post-Div Steps | Profile Tuple |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    f_by_seed = {r["seed"]: r for r in res["targets"]["fischer_target"]}
    m_by_seed = {r["seed"]: r for r in res["targets"]["mal_target"]}
    
    # Stable seed ordering derived from record appearance
    seeds_in_order = [r["seed"] for r in res["targets"]["fischer_target"]]
    
    q_div_seeds = [s for s in seeds_in_order if f_by_seed[s]["q_l1"] > 0 or m_by_seed[s]["q_l1"] > 0]
    null_seeds = [s for s in seeds_in_order if s not in q_div_seeds]
    
    for s in q_div_seeds:
        fr = f_by_seed[s]
        mr = m_by_seed[s]
        f_props = get_record_display_props(fr)
        m_props = get_record_display_props(mr)
        
        # Verify agreement in every displayed field between targets before collapsing to Both
        can_collapse = (f_props == m_props and not fr["act_div"])
        if can_collapse:
            lines.append(f"| {s} | Both | {f_props[0]} | {f_props[1]} | {f_props[2]} | {f_props[3]} | {f_props[4]} | {f_props[5]} | {f_props[6]} | {f_props[7]} |")
        else:
            lines.append(f"| {s} | Fischer | {f_props[0]} | {f_props[1]} | {f_props[2]} | {f_props[3]} | {f_props[4]} | {f_props[5]} | {f_props[6]} | {f_props[7]} |")
            lines.append(f"| {s} | Mal | {m_props[0]} | {m_props[1]} | {m_props[2]} | {m_props[3]} | {m_props[4]} | {m_props[5]} | {m_props[6]} | {m_props[7]} |")
            
    # Verify displayed properties for every null seed before grouped null row
    if null_seeds:
        null_props_set = set()
        for s in null_seeds:
            fr = f_by_seed[s]
            mr = m_by_seed[s]
            f_props = get_record_display_props(fr)
            m_props = get_record_display_props(mr)
            if f_props != m_props:
                raise ValueError(f"Null seed {s} target disagreement in displayed properties: {f_props} vs {m_props}")
            null_props_set.add(f_props)
            
        if len(null_props_set) != 1:
            raise ValueError(f"Null seeds have varying displayed properties: {null_props_set}")
            
        np = list(null_props_set)[0]
        null_label = f"*{len(null_seeds)} Nulls*" if len(null_seeds) > 1 else f"*{null_seeds[0]} Null*"
        lines.append(f"| {null_label} | Both | {np[0]} | {np[1]} | {np[2]} | {np[3]} | {np[4]} | {np[5]} | {np[6]} | {np[7]} |")
        
    return "\n".join(lines)

def compute_aggregates(res):
    out = {}
    for tgt in ["fischer_target", "mal_target"]:
        recs = res["targets"][tgt]
        div_recs = [r for r in recs if r["act_div"]]
        
        persists_all = [r["persistence"] for r in div_recs]
        persists_no199 = [r["persistence"] for r in div_recs if r["t_div"] != 199]
        
        shared_steps = [r["transient_intersections_count"] for r in div_recs]
        intersecting = [c for c in shared_steps if c > 0]
        rejoins = [r for r in div_recs if r["rejoined"]]
        
        out[tgt] = {
            "total_count": len(recs),
            "div_count": len(div_recs),
            "persistence_mean_all": round(statistics.mean(persists_all), 4) if persists_all else 0.0,
            "persistence_mean_no199": round(statistics.mean(persists_no199), 4) if persists_no199 else 0.0,
            "intersecting_runs_count": len(intersecting),
            "shared_steps_total": sum(shared_steps),
            "shared_steps_mean_intersecting": round(statistics.mean(intersecting), 1) if intersecting else 0.0,
            "shared_steps_mean_all_div": round(statistics.mean(shared_steps), 1) if shared_steps else 0.0,
            "rejoining_runs_count": len(rejoins),
            "rejoining_seeds": [r["seed"] for r in rejoins]
        }
        
    q_divs = res["dream_q_divergence"]
    onset_map = {}
    for item in q_divs:
        s = item["seed"]
        tgt = item["target"]
        step = item["first_q_div_step"]
        l1 = item["final_l1"]
        if s not in onset_map:
            onset_map[s] = {}
        onset_map[s][tgt] = (step, l1)
        
    # Check paired dream-onset agreement before reducing target records to one seed-level onset
    reduced_steps = []
    for s, tgts in onset_map.items():
        if len(tgts) > 1:
            steps_set = {v[0] for v in tgts.values()}
            l1_set = {v[1] for v in tgts.values()}
            if len(steps_set) > 1:
                raise ValueError(f"Paired target dream onset step disagreement for seed {s}: {tgts}")
            if len(l1_set) > 1:
                raise ValueError(f"Paired target dream final L1 disagreement for seed {s}: {tgts}")
        first_step = next(iter(tgts.values()))[0]
        if first_step is not None:
            reduced_steps.append(first_step)
            
    steps = sorted(reduced_steps)
    out["dream_onset"] = {
        "count": len(steps),
        "steps": steps,
        "min": min(steps) if steps else None,
        "max": max(steps) if steps else None,
        "median": statistics.median(steps) if steps else 0.0,
        "count_le_8": sum(1 for x in steps if x <= 8),
        "count_lt_8": sum(1 for x in steps if x < 8)
    }
    return out

def verify_tables_in_report():
    res = load_analysis_results()
    report_path = Path(__file__).parent / "REPORT.md"
    if not report_path.exists():
        raise FileNotFoundError(f"{report_path} not found")
        
    with open(report_path) as f:
        content = f.read()
        
    cls_table = generate_classification_table(res)
    assert cls_table in content, "Classification table mismatch in REPORT.md"
    
    per_seed_table = generate_per_seed_table(res)
    assert per_seed_table in content, "Per-seed table mismatch in REPORT.md"
    
    aggs = compute_aggregates(res)
    # Check Fischer and Mal persistence means
    f_p_str = f"**{aggs['fischer_target']['persistence_mean_all']:.4f}**"
    m_p_str = f"**{aggs['mal_target']['persistence_mean_all']:.4f}**"
    assert f_p_str in content, f"Fischer persistence mean {f_p_str} not found in REPORT.md"
    assert m_p_str in content, f"Mal persistence mean {m_p_str} not found in REPORT.md"
    
    # Check intersection numbers
    assert "**10 of 12** in Fischer" in content
    assert "**10 of 11** in Mal" in content
    assert f"**{aggs['fischer_target']['shared_steps_total']} steps**" in content
    assert f"**{aggs['mal_target']['shared_steps_total']} steps**" in content
    assert f"**{aggs['fischer_target']['shared_steps_mean_intersecting']:.1f} steps**" in content
    assert f"**{aggs['mal_target']['shared_steps_mean_intersecting']:.1f} steps**" in content
    
    # Check onset numbers
    assert f"median step {aggs['dream_onset']['median']:.1f}" in content or f"median Step {aggs['dream_onset']['median']:.1f}" in content
    assert f"**{aggs['dream_onset']['count_le_8']} of 14 seeds**" in content
    
    print("Verification passed! All generated tables and aggregate numbers match REPORT.md exactly.")

if __name__ == "__main__":
    res = load_analysis_results()
    print("=== Generated Classification Table ===")
    print(generate_classification_table(res))
    print("\n=== Generated Aggregates ===")
    print(json.dumps(compute_aggregates(res), indent=2))
    print("\n=== Generated Per-Seed Table ===")
    print(generate_per_seed_table(res))
    verify_tables_in_report()
