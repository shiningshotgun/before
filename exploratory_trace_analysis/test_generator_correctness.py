"""Regression and correctness tests for generate_report_tables.py.

Tests dynamic derivation and rigorous verification against REPORT.md, including:
1. Seed identity changes propagating to classification membership (catching hardcoded seed maps).
2. Record profile modifications propagating to counts, percentages, and irregular reporting.
3. Rejecting invalid 'Both' row collapse, varying null properties, and conflicting paired dream-onsets.
4. Calling the real verify_tables_in_report() verifier on in-memory report copies and record mutations:
   - unchanged report passes;
   - wrong aggregate value in designated block fails even when correct token appears elsewhere;
   - swapping target-specific aggregate values fails;
   - damaging or removing generated blocks fails;
   - mutating parsed aggregate records causes unchanged old block to fail and freshly rendered block to reflect new inputs.
"""
import copy
import pytest
from pathlib import Path

from exploratory_trace_analysis.generate_report_tables import (
    load_analysis_results,
    generate_classification_table,
    generate_per_seed_table,
    compute_aggregates,
    render_summary_metrics,
    render_onset_metrics,
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
    assert "12 (44.4%)" in table_text, "Fischer null count did not update dynamically to 12 (44.4%)"
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
    """Mutation Test 3: Must reject invalid 'Both' row collapse, varying nulls, and disagreeing paired dream onsets."""
    res = load_analysis_results()
    
    # 3a. Disagreeing displayed property for a seed eligible for Both
    mutated_props = copy.deepcopy(res)
    for r in mutated_props["targets"]["fischer_target"]:
        if r["seed"] == 1022:
            r["q_l1"] = 0.9999  # Disagrees with mal_target (0.6000)
            
    per_seed_text = generate_per_seed_table(mutated_props)
    assert "| 1022 | Both |" not in per_seed_text, "Seed 1022 collapsed to 'Both' despite conflicting q_l1"
    assert "| 1022 | Fischer | 0.9999 |" in per_seed_text, "Seed 1022 Fischer row missing or incorrect"
    assert "| 1022 | Mal | 0.6000 |" in per_seed_text, "Seed 1022 Mal row missing or incorrect"
    
    # 3b. Null seed variation
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

def test_real_verifier_unmodified_report_passes():
    """Verifier must pass 100% on unmodified report and unmodified records."""
    res = load_analysis_results()
    report_path = Path(__file__).parent / "REPORT.md"
    with open(report_path) as f:
        real_content = f.read()
    verify_tables_in_report(report_content=real_content, parsed_results=res)

def test_real_verifier_catches_wrong_location_with_token_elsewhere():
    """Mutation Test 4a: Wrong aggregate value in designated block fails even when correct token remains elsewhere."""
    res = load_analysis_results()
    report_path = Path(__file__).parent / "REPORT.md"
    with open(report_path) as f:
        real_content = f.read()
        
    # Replace Fischer persistence mean 0.3132 with 0.9999 inside the summary metrics block,
    # but place 0.3132 elsewhere on an unrelated line.
    corrupted_content = real_content.replace(
        "Fischer Target (12 divergent runs): Mean = **0.3132**",
        "Fischer Target (12 divergent runs): Mean = **0.9999**"
    )
    # Add dummy line with 0.3132 elsewhere in prose
    corrupted_content += "\n\nUnrelated note mentioning **0.3132** elsewhere.\n"
    
    with pytest.raises(AssertionError, match="Summary metrics block mismatch"):
        verify_tables_in_report(report_content=corrupted_content, parsed_results=res)

def test_real_verifier_catches_target_swapped_aggregates():
    """Mutation Test 4b: Swapping target-specific aggregate values fails verification."""
    res = load_analysis_results()
    report_path = Path(__file__).parent / "REPORT.md"
    with open(report_path) as f:
        real_content = f.read()
        
    # Swap Fischer and Mal persistence lines in the report
    f_line = "  - Fischer Target (12 divergent runs): Mean = **0.3132** (Min = 0.0000, Max = 1.0000)"
    m_line = "  - Mal Target (11 divergent runs): Mean = **0.2979** (Min = 0.0000, Max = 0.6983)"
    
    corrupted_content = real_content.replace(f_line, "__TEMP_LINE__").replace(m_line, f_line).replace("__TEMP_LINE__", m_line)
    with pytest.raises(AssertionError, match="Summary metrics block mismatch"):
        verify_tables_in_report(report_content=corrupted_content, parsed_results=res)

def test_real_verifier_catches_damaged_or_missing_delimiters():
    """Mutation Test 4c: Damaging or removing delimiter markers fails verification."""
    res = load_analysis_results()
    report_path = Path(__file__).parent / "REPORT.md"
    with open(report_path) as f:
        real_content = f.read()
        
    # Remove summary start marker
    corrupted_content = real_content.replace("<!-- START_GENERATED_SUMMARY_METRICS -->\n", "")
    with pytest.raises(AssertionError, match="Summary metrics delimiter markers missing"):
        verify_tables_in_report(report_content=corrupted_content, parsed_results=res)
        
    # Corrupt onset start marker
    corrupted_onset = real_content.replace("<!-- START_GENERATED_DIVERGENCE_ONSET -->\n", "")
    with pytest.raises(AssertionError, match="Divergence onset delimiter markers missing"):
        verify_tables_in_report(report_content=corrupted_onset, parsed_results=res)

def test_real_verifier_responds_to_parsed_record_mutations():
    """Mutation Test 4d: Mutating parsed aggregate records makes unchanged old block fail and fresh render reflect new inputs."""
    res = load_analysis_results()
    report_path = Path(__file__).parent / "REPORT.md"
    with open(report_path) as f:
        real_content = f.read()
        
    mutated_res = copy.deepcopy(res)
    # Mutate persistence of seed 1711 in fischer_target from 0.0571 to 0.9500
    for r in mutated_res["targets"]["fischer_target"]:
        if r["seed"] == 1711:
            r["persistence"] = 0.9500
            
    # Update the per-seed table in real_content copy so that verify_tables_in_report
    # tests specifically the summary metrics block against the mutated record
    old_per_seed = generate_per_seed_table(res)
    new_per_seed = generate_per_seed_table(mutated_res)
    report_copy = real_content.replace(old_per_seed, new_per_seed)
            
    # 1. Verification of the unchanged summary metrics block against mutated parsed records MUST fail
    with pytest.raises(AssertionError, match="Summary metrics block mismatch"):
        verify_tables_in_report(report_content=report_copy, parsed_results=mutated_res)
        
    # 2. Freshly rendered block over mutated records reflects the updated mean
    new_aggs = compute_aggregates(mutated_res)
    new_f_mean = new_aggs["fischer_target"]["persistence_mean_all"]
    freshly_rendered = render_summary_metrics(mutated_res, new_aggs)
    assert f"**{new_f_mean:.4f}**" in freshly_rendered
    assert "**0.3132**" not in freshly_rendered
