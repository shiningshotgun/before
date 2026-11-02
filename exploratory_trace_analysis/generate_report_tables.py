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
    
    seeds_in_order = [r["seed"] for r in res["targets"]["fischer_target"]]
    
    q_div_seeds = [s for s in seeds_in_order if f_by_seed[s]["q_l1"] > 0 or m_by_seed[s]["q_l1"] > 0]
    null_seeds = [s for s in seeds_in_order if s not in q_div_seeds]
    
    for s in q_div_seeds:
        fr = f_by_seed[s]
        mr = m_by_seed[s]
        f_props = get_record_display_props(fr)
        m_props = get_record_display_props(mr)
        
        can_collapse = (f_props == m_props and not fr["act_div"])
        if can_collapse:
            lines.append(f"| {s} | Both | {f_props[0]} | {f_props[1]} | {f_props[2]} | {f_props[3]} | {f_props[4]} | {f_props[5]} | {f_props[6]} | {f_props[7]} |")
        else:
            lines.append(f"| {s} | Fischer | {f_props[0]} | {f_props[1]} | {f_props[2]} | {f_props[3]} | {f_props[4]} | {f_props[5]} | {f_props[6]} | {f_props[7]} |")
            lines.append(f"| {s} | Mal | {m_props[0]} | {m_props[1]} | {m_props[2]} | {m_props[3]} | {m_props[4]} | {m_props[5]} | {m_props[6]} | {m_props[7]} |")
            
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

# Threshold for flagging horizon-edge rejoining runs in report bullets
HORIZON_EDGE_MATCHING_STEPS_THRESHOLD = 10  # Matching-step count strictly below this threshold receives 'horizon edge: ' label

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
        # Order rejoining records deterministically: ascending t_rejoin, ties broken by seed
        sorted_rejoins = sorted(rejoins, key=lambda r: (r["t_rejoin"], r["seed"]))
        
        out[tgt] = {
            "total_count": len(recs),
            "div_count": len(div_recs),
            "persistence_min": min(persists_all) if persists_all else 0.0,
            "persistence_max": max(persists_all) if persists_all else 0.0,
            "persistence_mean_all": round(statistics.mean(persists_all), 4) if persists_all else 0.0,
            "persistence_mean_no199": round(statistics.mean(persists_no199), 4) if persists_no199 else 0.0,
            "persistence_no199_count": len(persists_no199),
            "zero_horizon_seeds": [r["seed"] for r in div_recs if r["t_div"] == 199],
            "intersecting_runs_count": len(intersecting),
            "shared_steps_total": sum(shared_steps),
            "shared_steps_mean_intersecting": round(statistics.mean(intersecting), 1) if intersecting else 0.0,
            "shared_steps_mean_all_div": round(statistics.mean(shared_steps), 1) if shared_steps else 0.0,
            "rejoining_runs_count": len(rejoins),
            "rejoining_seeds": [r["seed"] for r in sorted_rejoins]
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
        
    reduced_steps = []
    reduced_l1 = []
    for s, tgts in onset_map.items():
        if len(tgts) > 1:
            steps_set = {v[0] for v in tgts.values()}
            l1_set = {v[1] for v in tgts.values()}
            if len(steps_set) > 1:
                raise ValueError(f"Paired target dream onset step disagreement for seed {s}: {tgts}")
            if len(l1_set) > 1:
                raise ValueError(f"Paired target dream final L1 disagreement for seed {s}: {tgts}")
        first_step = next(iter(tgts.values()))[0]
        final_l1 = next(iter(tgts.values()))[1]
        if first_step is not None:
            reduced_steps.append(first_step)
        if final_l1 is not None and final_l1 > 0:
            reduced_l1.append(final_l1)
            
    steps = sorted(reduced_steps)
    min_l1 = min(reduced_l1) if reduced_l1 else 0.0
    max_l1 = max(reduced_l1) if reduced_l1 else 0.0
    
    distinct_l1 = sorted(list(set(reduced_l1)))
    examples = []
    for v in distinct_l1:
        is_mult = abs(v / 0.3 - round(v / 0.3)) < 1e-9
        is_endpoint = (abs(v - min_l1) < 1e-9 or abs(v - max_l1) < 1e-9)
        if not is_mult and not is_endpoint:
            examples.append(f"{v:.4f}")
            
    out["dream_onset"] = {
        "count": len(steps),
        "steps": steps,
        "min": min(steps) if steps else None,
        "max": max(steps) if steps else None,
        "median": statistics.median(steps) if steps else 0.0,
        "count_le_8": sum(1 for x in steps if x <= 8),
        "count_lt_8": sum(1 for x in steps if x < 8),
        "l1_min": min_l1,
        "l1_max": max_l1,
        "l1_examples": examples
    }
    return out

def render_summary_metrics(res, aggs=None):
    """Deterministically renders the Section 4 Summary Metrics block."""
    if aggs is None:
        aggs = compute_aggregates(res)
        
    f_agg = aggs["fischer_target"]
    m_agg = aggs["mal_target"]
    f_recs = [r for r in res["targets"]["fischer_target"] if r["act_div"]]
    m_recs = [r for r in res["targets"]["mal_target"] if r["act_div"]]
    
    f_zero_inter = [r["seed"] for r in f_recs if r["transient_intersections_count"] == 0]
    m_zero_inter = [r["seed"] for r in m_recs if r["transient_intersections_count"] == 0]
    f_zero_str = " and ".join(str(s) for s in f_zero_inter)
    m_zero_str = " and ".join(str(s) for s in m_zero_inter)
    
    f_div_count = f_agg["div_count"]
    m_div_count = m_agg["div_count"]
    f_p_all = f_agg["persistence_mean_all"]
    m_p_all = m_agg["persistence_mean_all"]
    f_p_min = f_agg["persistence_min"]
    f_p_max = f_agg["persistence_max"]
    m_p_min = m_agg["persistence_min"]
    m_p_max = m_agg["persistence_max"]
    
    f_p_no199 = f_agg["persistence_mean_no199"]
    m_p_no199 = m_agg["persistence_mean_no199"]
    f_no199_count = f_agg["persistence_no199_count"]
    m_no199_count = m_agg["persistence_no199_count"]
    
    # Zero-horizon seed identity and clauses
    f_zh = f_agg["zero_horizon_seeds"]
    m_zh = f_agg["zero_horizon_seeds"]
    if f_zh == m_zh:
        if len(f_zh) == 1:
            zh_clause = "treating seed " + str(f_zh[0]) + " ($t_{\\text{div}}=199$, leaving zero post-divergence steps remaining in the 200-step horizon) as having $P_{\\text{state}} = 0.0000$ identically for both targets"
            zh_excl_clause = "If seed " + str(f_zh[0]) + " is excluded due to an empty post-divergence horizon"
        elif len(f_zh) > 1:
            seeds_str = ", ".join(str(s) for s in f_zh)
            zh_clause = "treating seeds " + seeds_str + " ($t_{\\text{div}}=199$, leaving zero post-divergence steps remaining in the 200-step horizon) as having $P_{\\text{state}} = 0.0000$ identically for both targets"
            zh_excl_clause = "If seeds " + seeds_str + " are excluded due to an empty post-divergence horizon"
        else:
            zh_clause = "with all divergent runs having non-empty post-divergence horizons"
            zh_excl_clause = "With no seeds excluded due to an empty post-divergence horizon"
    else:
        zh_clause = "treating zero-horizon seeds (Fischer: " + str(f_zh) + ", Mal: " + str(m_zh) + ") as having $P_{\\text{state}} = 0.0000$"
        zh_excl_clause = "If zero-horizon seeds (Fischer: " + str(f_zh) + ", Mal: " + str(m_zh) + ") are excluded"
        
    f_inter_count = f_agg["intersecting_runs_count"]
    m_inter_count = m_agg["intersecting_runs_count"]
    f_shared_total = f_agg["shared_steps_total"]
    m_shared_total = m_agg["shared_steps_total"]
    f_shared_mean_inter = f_agg["shared_steps_mean_intersecting"]
    m_shared_mean_inter = m_agg["shared_steps_mean_intersecting"]
    f_shared_mean_all = f_agg["shared_steps_mean_all_div"]
    m_shared_mean_all = m_agg["shared_steps_mean_all_div"]
    f_rej_count = f_agg["rejoining_runs_count"]
    m_rej_count = m_agg["rejoining_runs_count"]
    
    lines = []
    lines.append("- **State Persistence ($P_{\\text{state}}$):** Fraction of post-divergence steps ($t > t_{\\text{div}}$) spent in different states.")
    lines.append("  *Uniform Aggregation Rule:* Includes all action-divergent runs ($N=" + str(f_div_count) + "$ in Fischer, $N=" + str(m_div_count) + "$ in Mal), " + zh_clause + ":")
    lines.append("  - Fischer Target (" + str(f_div_count) + " divergent runs): Mean = **" + f"{f_p_all:.4f}" + "** (Min = " + f"{f_p_min:.4f}" + ", Max = " + f"{f_p_max:.4f}" + ")")
    lines.append("  - Mal Target (" + str(m_div_count) + " divergent runs): Mean = **" + f"{m_p_all:.4f}" + "** (Min = " + f"{m_p_min:.4f}" + ", Max = " + f"{m_p_max:.4f}" + ")")
    lines.append("  *(" + zh_excl_clause + ", the means across non-empty horizons are " + f"{f_p_no199:.4f}" + " over " + str(f_no199_count) + " Fischer runs and " + f"{m_p_no199:.4f}" + " over " + str(m_no199_count) + " Mal runs; the previous draft inadvertently reported the 10-run Mal mean alongside the 12-run Fischer mean)*.")
    lines.append("- **Transient State Intersections:**")
    lines.append("  The analysis records full-state matches (both grid position and collected object set identical), not merely grid coordinates.")
    lines.append("  - Runs with at least one post-divergence state intersection: **" + str(f_inter_count) + " of " + str(f_div_count) + "** in Fischer (seeds " + f_zero_str + " have none) and **" + str(m_inter_count) + " of " + str(m_div_count) + "** in Mal (seed " + m_zero_str + " has none).")
    lines.append("  - Total shared post-divergence steps across all runs: **" + str(f_shared_total) + " steps** (Fischer) and **" + str(m_shared_total) + " steps** (Mal).")
    lines.append("  - Means over intersecting runs ($N=" + str(f_inter_count) + "$): Mean = **" + f"{f_shared_mean_inter:.1f}" + " steps** (Fischer) and **" + f"{m_shared_mean_inter:.1f}" + " steps** (Mal).")
    lines.append("  *(Across all divergent runs including zero-intersection runs, the means are " + f"{f_shared_mean_all:.1f}" + " steps over " + str(f_div_count) + " Fischer runs and " + f"{m_shared_mean_all:.1f}" + " steps over " + str(m_div_count) + " Mal runs)*.")
    lines.append("- **Absorbing Rejoining:**")
    lines.append("  An absorbing rejoin is defined as the existence of a step $t_{\\text{rejoin}}$ such that for all $t \\in [t_{\\text{rejoin}}, 199]$, both state and action match through the end of the logged return trace.")
    
    # Rejoining bullets derived dynamically: ascending t_rejoin, ties broken by seed
    for tgt_label, recs, count, div_c in [
        ("Fischer Target", f_recs, f_rej_count, f_div_count),
        ("Mal Target", m_recs, m_rej_count, m_div_count)
    ]:
        lines.append("  - **" + tgt_label + ":** " + str(count) + " of " + str(div_c) + " action-divergent runs absorbed/rejoined through step 199:")
        rejoins = [r for r in recs if r["rejoined"]]
        sorted_rejoins = sorted(rejoins, key=lambda r: (r["t_rejoin"], r["seed"]))
        for r in sorted_rejoins:
            s = r["seed"]
            t_d = r["t_div"]
            t_r = r["t_rejoin"]
            m_s = 200 - t_r
            prefix = "horizon edge: " if m_s < HORIZON_EDGE_MATCHING_STEPS_THRESHOLD else ""
            lines.append(f"    - Seed `{s}`: diverged at $t={t_d}$, absorbed at $t={t_r}$ ({prefix}{m_s} matching steps through step 199).")
            
    return "\n".join(lines)

def render_onset_metrics(aggs):
    """Deterministically renders the Section 6 Divergence Onset Metrics block."""
    onset = aggs["dream_onset"]
    o_count = onset["count"]
    o_min = onset["min"]
    o_max = onset["max"]
    o_med = onset["median"]
    o_le_8 = onset["count_le_8"]
    o_lt_8 = onset["count_lt_8"]
    l1_min = onset["l1_min"]
    l1_max = onset["l1_max"]
    examples_str = ", ".join(onset["l1_examples"])

    lines = []
    lines.append("Among the " + str(o_count) + " seeds exhibiting post-dream Q divergence ($L_1 > 0$):")
    lines.append("- Q-table divergence first manifests early in the dream phase:")
    lines.append("  - Range of first divergent update: **Step " + str(o_min) + " to Step " + str(o_max) + "**, with **median step " + f"{o_med:.1f}" + "**.")
    lines.append("  - **" + str(o_le_8) + " of " + str(o_count) + " seeds** exhibit onset at or before update 8 (and **" + str(o_lt_8) + " of " + str(o_count) + "** strictly before update 8).")
    lines.append("- Total final Q $L_1$ distances range from " + f"{l1_min:.4f}" + " to " + f"{l1_max:.4f}" + ". Because TD error updates incorporate discount factor $\\gamma = 0.95$ and non-zero next-state values, compounding over successive updates yields values (e.g., " + examples_str + ") that are not simple integer multiples of the learning rate ($\\alpha = 0.3$).")
    return "\n".join(lines)


def verify_tables_in_report(report_content=None, parsed_results=None):
    """Verifies that REPORT.md tables and delimited aggregate blocks match derived outputs exactly.
    
    Accepts report_content string and parsed_results dict optionally for in-memory testing.
    """
    if parsed_results is None:
        res = load_analysis_results()
    else:
        res = parsed_results
        
    if report_content is None:
        report_path = Path(__file__).parent / "REPORT.md"
        if not report_path.exists():
            raise FileNotFoundError(f"{report_path} not found")
        with open(report_path) as f:
            content = f.read()
    else:
        content = report_content
        
    cls_table = generate_classification_table(res)
    assert cls_table in content, "Classification table mismatch in REPORT.md"
    
    per_seed_table = generate_per_seed_table(res)
    assert per_seed_table in content, "Per-seed table mismatch in REPORT.md"
    
    aggs = compute_aggregates(res)
    
    # 1. Exact comparison of Section 4 Summary Metrics delimited block
    marker_s_start = "<!-- START_GENERATED_SUMMARY_METRICS -->"
    marker_s_end = "<!-- END_GENERATED_SUMMARY_METRICS -->"
    if marker_s_start not in content or marker_s_end not in content:
        raise AssertionError("Summary metrics delimiter markers missing in REPORT.md")
        
    actual_summary = content.split(marker_s_start + "\n")[1].split("\n" + marker_s_end)[0]
    expected_summary = render_summary_metrics(res, aggs)
    assert actual_summary == expected_summary, f"Summary metrics block mismatch in REPORT.md:\n--- ACTUAL ---\n{actual_summary}\n--- EXPECTED ---\n{expected_summary}"

    # 2. Exact comparison of Section 6 Divergence Onset Metrics delimited block
    marker_o_start = "<!-- START_GENERATED_DIVERGENCE_ONSET -->"
    marker_o_end = "<!-- END_GENERATED_DIVERGENCE_ONSET -->"
    if marker_o_start not in content or marker_o_end not in content:
        raise AssertionError("Divergence onset delimiter markers missing in REPORT.md")
        
    actual_onset = content.split(marker_o_start + "\n")[1].split("\n" + marker_o_end)[0]
    expected_onset = render_onset_metrics(aggs)
    assert actual_onset == expected_onset, f"Divergence onset block mismatch in REPORT.md:\n--- ACTUAL ---\n{actual_onset}\n--- EXPECTED ---\n{expected_onset}"
    
    print("Verification passed! All generated tables and aggregate blocks match REPORT.md exactly.")

if __name__ == "__main__":
    res = load_analysis_results()
    print("=== Generated Classification Table ===")
    print(generate_classification_table(res))
    print("\n=== Generated Aggregates ===")
    print(json.dumps(compute_aggregates(res), indent=2))
    print("\n=== Generated Summary Metrics Block ===")
    print(render_summary_metrics(res))
    print("\n=== Generated Onset Metrics Block ===")
    print(render_onset_metrics(compute_aggregates(res)))
    print("\n=== Generated Per-Seed Table ===")
    print(generate_per_seed_table(res))
    verify_tables_in_report()
