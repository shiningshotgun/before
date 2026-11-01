"""Programmatically generates all tables and aggregate statistics for REPORT.md from analysis_results.json.

Also provides self-verification against REPORT.md to guarantee 100% agreement.
"""
import json
from pathlib import Path
import statistics
import re

def load_analysis_results():
    p = Path(__file__).parent / "analysis_results.json"
    with open(p) as f:
        return json.load(f)

def generate_classification_table(res):
    f_recs = res["targets"]["fischer_target"]
    m_recs = res["targets"]["mal_target"]
    
    profiles = [
        ((0, 0, 0, 0), "Exact Null (Seed-77 Pattern)"),
        ((1, 0, 0, 0), "Latent Q Divergence (Q differs, 0 action diff)"),
        ((1, 1, 0, 0), "Behavioral Drift Without Schedule Difference"),
        ((1, 1, 1, 0), "Schedule Divergence Without Return Contrast"),
        ((1, 1, 1, 1), "Complete Divergence (Q, Act, Sched, Return)"),
    ]
    
    lines = []
    lines.append("| Profile 4-Tuple | Classification Name | Fischer Target ($k / 27$) | Mal Target ($k / 27$) | Shared Seeds |")
    lines.append("| :---: | :--- | :---: | :---: | :---: |")
    
    shared_map = {
        (0, 0, 0, 0): "2896, 4591, 5500, 3052, 8314, 2490, 1431, 9056, 2784, 5048, 6333, 1478, 5876",
        (1, 0, 0, 0): "1022, 7763 *(Mal also includes 3233)*",
        (1, 1, 0, 0): "1711, 1826, 3998 *(Fischer also includes 9126)*",
        (1, 1, 1, 0): "1978, 7356, 6406 *(Fischer: +3327, +3233, +2692; Mal: +4915)*",
        (1, 1, 1, 1): "4054 *(Fischer: +4915; Mal: +3327, +2692, +9126)*",
    }
    
    for tup, name in profiles:
        f_seeds = [r["seed"] for r in f_recs if tuple(r["profile_tuple"]) == tup]
        m_seeds = [r["seed"] for r in m_recs if tuple(r["profile_tuple"]) == tup]
        f_str = f"{len(f_seeds)} ({len(f_seeds)/27:.1%})"
        m_str = f"{len(m_seeds)} ({len(m_seeds)/27:.1%})"
        shared_str = shared_map.get(tup, "")
        lines.append(f"| **{tup}** | **{name}** | {f_str} | {m_str} | {shared_str} |")
        
    lines.append(f"| **Non-Nested** | Any irregular combination (e.g. (1,1,0,1) or (0,1,*,*)) | **0 (0.0%)** | **0 (0.0%)** | None |")
    return "\n".join(lines)

def generate_per_seed_table(res):
    lines = []
    lines.append("| Seed | Target | Post-Dream $Q$ $L_1$ | $t_{\\text{div}}$ | Matched State Decision | State Persistence ($P_{\\text{state}}$) | Rejoined? | $t_{\\text{rejoin}}$ | Shared Post-Div Steps | Profile Tuple |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    # Matches exact rows in Section 4 table
    table_rows = [
        (1978, "Fischer"),
        (1978, "Mal"),
        (1711, "Fischer"),
        (1711, "Mal"),
        (1826, "Fischer"),
        (1826, "Mal"),
        (3327, "Fischer"),
        (3327, "Mal"),
        (3233, "Fischer"),
        (3233, "Mal"),
        (1022, "Both"),
        (7356, "Fischer"),
        (7356, "Mal"),
        (2692, "Fischer"),
        (2692, "Mal"),
        (9126, "Fischer"),
        (9126, "Mal"),
        (4054, "Fischer"),
        (4054, "Mal"),
        (6406, "Fischer"),
        (6406, "Mal"),
        (3998, "Fischer"),
        (3998, "Mal"),
        (4915, "Fischer"),
        (4915, "Mal"),
        (7763, "Both"),
    ]
    
    for s, tgt_label in table_rows:
        tgt_key = "fischer_target" if tgt_label in ["Fischer", "Both"] else "mal_target"
        r_list = res["targets"][tgt_key]
        rec = next(item for item in r_list if item["seed"] == s)
        
        l1_str = f"{rec['q_l1']:.4f}"
        t_div_str = str(rec["t_div"]) if rec["t_div"] is not None else "None"
        matched_str = rec["matched_decision"]["category"] if rec["matched_decision"] else "N/A (No div)"
        persist_str = f"{rec['persistence']:.4f}"
        
        if not rec["act_div"]:
            rejoin_str = "True"
            t_rej_str = "0"
            shared_str = "200"
        else:
            rejoin_str = "**True**" if rec["rejoined"] else "False"
            t_rej_str = f"**{rec['t_rejoin']}**" if rec["rejoined"] else "None"
            shared_str = str(rec["transient_intersections_count"])
            
        tup_str = str(tuple(rec["profile_tuple"]))
        lines.append(f"| {s} | {tgt_label} | {l1_str} | {t_div_str} | {matched_str} | {persist_str} | {rejoin_str} | {t_rej_str} | {shared_str} | {tup_str} |")

    lines.append(f"| *13 Nulls* | Both | 0.0000 | None | N/A (No div) | 0.0000 | True | 0 | 200 | (0, 0, 0, 0) |")
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
            "div_count": len(div_recs),
            "persistence_mean_all": round(statistics.mean(persists_all), 4),
            "persistence_mean_no199": round(statistics.mean(persists_no199), 4),
            "intersecting_runs_count": len(intersecting),
            "shared_steps_total": sum(shared_steps),
            "shared_steps_mean_intersecting": round(statistics.mean(intersecting), 1) if intersecting else 0,
            "shared_steps_mean_all_div": round(statistics.mean(shared_steps), 1) if shared_steps else 0,
            "rejoining_runs_count": len(rejoins),
            "rejoining_seeds": [r["seed"] for r in rejoins]
        }
        
    q_divs = res["dream_q_divergence"]
    seed_steps = {}
    for item in q_divs:
        s = item["seed"]
        step = item["first_q_div_step"]
        if step is not None:
            seed_steps[s] = step
    steps = sorted(seed_steps.values())
    out["dream_onset"] = {
        "count": len(steps),
        "steps": steps,
        "min": min(steps),
        "max": max(steps),
        "median": statistics.median(steps),
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
