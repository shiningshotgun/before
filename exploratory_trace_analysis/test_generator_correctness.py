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

def test_rejoining_identity_derived_from_records():
    """Test 1: Renaming rejoining seed 1711 to 99999 updates bullets and fails stale verification."""
    res = load_analysis_results()
    report_path = Path(__file__).parent / "REPORT.md"
    with open(report_path) as f:
        real_content = f.read()

    mutated = copy.deepcopy(res)
    # Rename seed 1711 to 99999 in targets and dream_q_divergence
    for r in mutated["targets"]["fischer_target"]:
        if r["seed"] == 1711:
            r["seed"] = 99999
    for r in mutated["targets"]["mal_target"]:
        if r["seed"] == 1711:
            r["seed"] = 99999
    for item in mutated["dream_q_divergence"]:
        if item["seed"] == 1711:
            item["seed"] = 99999

    # Fresh render must list 99999 and must not list 1711
    rendered = render_summary_metrics(mutated)
    assert "Seed `99999`:" in rendered
    assert "Seed `1711`:" not in rendered

    # Update classification table and per-seed table in real_content copy so test isolates summary metrics block verification failure
    old_cls = generate_classification_table(res)
    new_cls = generate_classification_table(mutated)
    old_per_seed = generate_per_seed_table(res)
    new_per_seed = generate_per_seed_table(mutated)
    report_copy = real_content.replace(old_cls, new_cls).replace(old_per_seed, new_per_seed)

    # verify_tables_in_report must fail against mutated records when summary block is unmutated
    with pytest.raises(AssertionError, match="Summary metrics block mismatch"):
        verify_tables_in_report(report_content=report_copy, parsed_results=mutated)

def test_persistence_max_derived_from_records():
    """Test 2: Mutating Mal divergent persistence max to 0.9500 updates render and fails stale verifier."""
    res = load_analysis_results()
    report_path = Path(__file__).parent / "REPORT.md"
    with open(report_path) as f:
        real_content = f.read()

    mutated = copy.deepcopy(res)
    # Set one Mal divergent record's persistence to 0.9500
    for r in mutated["targets"]["mal_target"]:
        if r["seed"] == 1711:
            r["persistence"] = 0.9500

    rendered = render_summary_metrics(mutated)
    assert "Max = 0.9500" in rendered
    assert "Max = 0.6983" not in rendered

    old_per_seed = generate_per_seed_table(res)
    new_per_seed = generate_per_seed_table(mutated)
    report_copy = real_content.replace(old_per_seed, new_per_seed)

    with pytest.raises(AssertionError, match="Summary metrics block mismatch"):
        verify_tables_in_report(report_content=report_copy, parsed_results=mutated)

def test_zero_horizon_identity_and_denominators():
    """Test 3: Zero-horizon seed identity and exclusion denominators derived from records."""
    res = load_analysis_results()

    # Part a: Move t_div == 199 from 1826 to a different divergent seed (3998)
    mut_a = copy.deepcopy(res)
    for tgt in ["fischer_target", "mal_target"]:
        for r in mut_a["targets"][tgt]:
            if r["seed"] == 1826:
                r["t_div"] = 50
                r["persistence"] = 0.5000
            elif r["seed"] == 3998:
                r["t_div"] = 199
                r["persistence"] = 0.0000

    rendered_a = render_summary_metrics(mut_a)
    lines_a = rendered_a.splitlines()
    # The first line of State Persistence rule and the exclusion note must name 3998 and not 1826
    rule_lines = [l for l in lines_a if "treating seed" in l or "excluded due to an empty" in l]
    assert len(rule_lines) == 2
    assert all("3998" in l for l in rule_lines)
    assert all("1826" not in l for l in rule_lines)

    # Part b: Additionally set a second divergent seed's t_div to 199 in both targets
    mut_b = copy.deepcopy(mut_a)
    for tgt in ["fischer_target", "mal_target"]:
        for r in mut_b["targets"][tgt]:
            if r["seed"] == 7356:
                r["t_div"] = 199
                r["persistence"] = 0.0000

    rendered_b = render_summary_metrics(mut_b)
    # Both zero-horizon seeds must be named and denominators drop to 10 and 9
    assert "3998" in rendered_b and "7356" in rendered_b
    assert "over 10 Fischer runs and" in rendered_b
    assert "over 9 Mal runs" in rendered_b

def test_onset_range_and_examples_derivation():
    """Test 4: Onset final L1 range and example list derived dynamically."""
    res = load_analysis_results()
    report_path = Path(__file__).parent / "REPORT.md"
    with open(report_path) as f:
        real_content = f.read()

    # Part a: Endpoint mutation - set seed 9126 final_l1 to 9.0 in both targets
    mut_a = copy.deepcopy(res)
    for item in mut_a["dream_q_divergence"]:
        if item["seed"] == 9126:
            item["final_l1"] = 9.0
    for tgt in ["fischer_target", "mal_target"]:
        for r in mut_a["targets"][tgt]:
            if r["seed"] == 9126:
                r["q_l1"] = 9.0

    aggs_a = compute_aggregates(mut_a)
    rendered_onset_a = render_onset_metrics(aggs_a)
    assert "range from 0.3000 to 9.0000" in rendered_onset_a
    # Example list must be exactly the current seven values: 9.0 excluded as endpoint and multiple of 0.3
    assert "(e.g., 0.3187, 0.6006, 0.6016, 0.6540, 0.9062, 0.9237, 1.2078)" in rendered_onset_a

    # Update per-seed table in real_content copy so test isolates onset block mismatch
    old_per_seed = generate_per_seed_table(res)
    new_per_seed_a = generate_per_seed_table(mut_a)
    report_copy_a = real_content.replace(old_per_seed, new_per_seed_a)

    with pytest.raises(AssertionError, match="Divergence onset block mismatch"):
        verify_tables_in_report(report_content=report_copy_a, parsed_results=mut_a)

    # Part b: Example inclusion - set seed 1022 final_l1 to 0.7777 in both targets
    mut_b = copy.deepcopy(res)
    for item in mut_b["dream_q_divergence"]:
        if item["seed"] == 1022:
            item["final_l1"] = 0.7777
    for tgt in ["fischer_target", "mal_target"]:
        for r in mut_b["targets"][tgt]:
            if r["seed"] == 1022:
                r["q_l1"] = 0.7777

    aggs_b = compute_aggregates(mut_b)
    rendered_onset_b = render_onset_metrics(aggs_b)
    assert "range from 0.3000 to 1.5122" in rendered_onset_b
    assert "(e.g., 0.3187, 0.6006, 0.6016, 0.6540, 0.7777, 0.9062, 0.9237, 1.2078)" in rendered_onset_b

    new_per_seed_b = generate_per_seed_table(mut_b)
    report_copy_b = real_content.replace(old_per_seed, new_per_seed_b)

    with pytest.raises(AssertionError, match="Divergence onset block mismatch"):
        verify_tables_in_report(report_content=report_copy_b, parsed_results=mut_b)

def test_horizon_edge_threshold_derivation():
    """Test 5: Setting a rejoined seed's t_rejoin below threshold adds horizon edge prefix with derived count."""
    res = load_analysis_results()
    mut = copy.deepcopy(res)

    # In Fischer target, seed 4915 had t_rejoin=175 (matching steps = 25 >= 10).
    # Move its t_rejoin to 195 (matching steps = 5 < 10 threshold).
    for r in mut["targets"]["fischer_target"]:
        if r["seed"] == 4915:
            r["t_rejoin"] = 195

    rendered = render_summary_metrics(mut)
    assert "Seed `4915`: diverged at $t=63$, absorbed at $t=195$ (horizon edge: 5 matching steps through step 199)." in rendered
