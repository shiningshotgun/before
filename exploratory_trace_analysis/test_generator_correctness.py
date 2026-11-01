"""Regression and correctness tests for generate_report_tables.py.

Tests dynamic derivation from in-memory parsed records, including:
1. Seed identity changes propagating to classification membership (catching hardcoded seed maps).
2. Record profile modifications propagating to counts, percentages, and irregular reporting.
3. Rejecting invalid 'Both' row collapse and conflicting paired dream-onsets.
4. Catching corrupted table cells or aggregates against report content.
"""
import copy
import pytest
from pathlib import Path

from exploratory_trace_analysis.generate_report_tables import (
    load_analysis_results,
    generate_classification_table,
    generate_per_seed_table,
    compute_aggregates,
    verify_tables_in_report
)

def test_seed_identity_mutation_propagates_to_classification():
    """Mutation Test 1: Renaming a seed in both targets must change generated classification text."""
    res = load_analysis_results()
    mutated = copy.deepcopy(res)
    
    # Replace seed 2896 with 99999 in both targets
    found_f = False
    for r in mutated["targets"]["fischer_target"]:
        if r["seed"] == 2896:
            r["seed"] = 99999
            found_f = True
    found_m = False
    for r in mutated["targets"]["mal_target"]:
        if r["seed"] == 2896:
            r["seed"] = 99999
            found_m = True
            
    assert found_f and found_m, "Seed 2896 not found in original data"
    
    table_text = generate_classification_table(mutated)
    # Must contain the new seed identity and MUST NOT contain the old seed in the null row
    assert "99999" in table_text, "Mutated seed 99999 did not appear in classification table"
    assert "2896" not in table_text, "Old seed 2896 unexpectedly remained in classification table (hardcoded mapping failure)"

def test_profile_mutation_propagates_to_counts_and_irregular_reporting():
    """Mutation Test 2: Changing profile indicators changes counts, percentages, and reports non-nested rows."""
    res = load_analysis_results()
    mutated = copy.deepcopy(res)
    
    # Mutate seed 2896 from (0,0,0,0) to irregular non-nested profile (1, 0, 1, 0)
    for r in mutated["targets"]["fischer_target"]:
        if r["seed"] == 2896:
            r["profile_tuple"] = [1, 0, 1, 0]
            
    table_text = generate_classification_table(mutated)
    # The null row count in Fischer should decrease from 13 to 12
    assert "12 (44.4%)" in table_text, "Fischer null count did not update dynamically to 12 (44.4%)"
    # An irregular row for (1, 0, 1, 0) must appear
    assert "(1, 0, 1, 0)" in table_text, "Irregular profile (1, 0, 1, 0) was not reported"
    assert "2896" in table_text, "Mutated seed 2896 was not identified in irregular profile row"

    # Test non-standard denominator
    mutated_subset = copy.deepcopy(res)
    mutated_subset["targets"]["fischer_target"] = mutated_subset["targets"]["fischer_target"][:20]
    mutated_subset["targets"]["mal_target"] = mutated_subset["targets"]["mal_target"][:20]
    table_subset = generate_classification_table(mutated_subset)
    assert "Fischer Target ($k / 20$)" in table_subset, "Table header did not derive actual record denominator"
    assert "Mal Target ($k / 20$)" in table_subset, "Table header did not derive actual record denominator"

def test_collapse_and_paired_onset_integrity():
    """Mutation Test 3: Must reject invalid 'Both' row collapse and disagreeing paired dream onsets."""
    res = load_analysis_results()
    
    # 3a. Disagreeing displayed property for a seed eligible for Both
    mutated_props = copy.deepcopy(res)
    for r in mutated_props["targets"]["fischer_target"]:
        if r["seed"] == 1022:
            r["q_l1"] = 0.9999  # Disagrees with mal_target (0.6000)
            
    per_seed_text = generate_per_seed_table(mutated_props)
    # Because q_l1 now differs, 1022 cannot collapse to Both; it must split into Fischer and Mal rows
    assert "| 1022 | Both |" not in per_seed_text, "Seed 1022 collapsed to 'Both' despite conflicting q_l1"
    assert "| 1022 | Fischer | 0.9999 |" in per_seed_text, "Seed 1022 Fischer row missing or incorrect"
    assert "| 1022 | Mal | 0.6000 |" in per_seed_text, "Seed 1022 Mal row missing or incorrect"
    
    # 3b. Null seed variation (e.g. changing profile tuple of one null seed without changing q_l1)
    mutated_null = copy.deepcopy(res)
    for r in mutated_null["targets"]["fischer_target"]:
        if r["seed"] == 2896:
            r["profile_tuple"] = [0, 0, 1, 0]
    for r in mutated_null["targets"]["mal_target"]:
        if r["seed"] == 2896:
            r["profile_tuple"] = [0, 0, 1, 0]
            
    with pytest.raises(ValueError, match="varying displayed properties"):
        generate_per_seed_table(mutated_null)

    # 3c. Conflicting paired dream-onset values
    mutated_onset = copy.deepcopy(res)
    for item in mutated_onset["dream_q_divergence"]:
        if item["seed"] == 1978 and item["target"] == "mal_target":
            item["first_q_div_step"] = 99  # Disagrees with fischer_target (1)
            
    with pytest.raises(ValueError, match="Paired target dream onset step disagreement"):
        compute_aggregates(mutated_onset)

def test_corrupted_report_comparison_fails():
    """Mutation Test 4: Corrupting a table cell or aggregate in REPORT copy must fail verification."""
    report_path = Path(__file__).parent / "REPORT.md"
    with open(report_path) as f:
        real_content = f.read()
        
    # Corrupt a single number in classification table (e.g., change 48.1% to 49.9%)
    corrupted_cls = real_content.replace("13 (48.1%)", "13 (49.9%)", 1)
    res = load_analysis_results()
    cls_table = generate_classification_table(res)
    assert cls_table not in corrupted_cls, "Corrupted classification table was not detected"

    # Corrupt an aggregate metric (e.g., change persistence mean 0.3132 to 0.9999)
    corrupted_agg = real_content.replace("**0.3132**", "**0.9999**")
    aggs = compute_aggregates(res)
    f_p_str = f"**{aggs['fischer_target']['persistence_mean_all']:.4f}**"
    assert f_p_str not in corrupted_agg, "Corrupted persistence aggregate was not detected"

def test_unmodified_report_verification_passes():
    """Unchanged report and data must pass self-verification 100%."""
    verify_tables_in_report()
