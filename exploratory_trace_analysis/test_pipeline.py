"""Tests verifying the full Stage 3 analysis pipeline outputs."""
import json
from pathlib import Path
import pytest
from exploratory_trace_analysis.run_analysis import run_full_analysis

def test_pipeline_integrity():
    res = run_full_analysis()
    
    assert "targets" in res
    assert "fischer_target" in res["targets"]
    assert "mal_target" in res["targets"]
    
    f_recs = res["targets"]["fischer_target"]
    m_recs = res["targets"]["mal_target"]
    
    assert len(f_recs) == 27
    assert len(m_recs) == 27
    
    # Check that matched decision is categorized for every action-divergent run
    for rec in f_recs:
        if rec["act_div"]:
            assert rec["matched_decision"] is not None
            assert rec["matched_decision"]["category"] in ["greedy_value_shift", "random_exploration_branch", "tie_breaking", "insufficient_data"]
            assert rec["matched_decision"]["category"] == "greedy_value_shift"
            
    for rec in m_recs:
        if rec["act_div"]:
            assert rec["matched_decision"] is not None
            assert rec["matched_decision"]["category"] in ["greedy_value_shift", "random_exploration_branch", "tie_breaking", "insufficient_data"]
            assert rec["matched_decision"]["category"] == "greedy_value_shift"
            
    # Check case studies
    assert "3233" in res["case_studies"]
    assert "9126" in res["case_studies"]
    
    cs_3233 = res["case_studies"]["3233"]
    assert cs_3233["targets"]["fischer_target"]["sched_d"] != cs_3233["targets"]["fischer_target"]["sched_u"]
    assert cs_3233["targets"]["mal_target"]["sched_d"] == cs_3233["targets"]["mal_target"]["sched_u"]
    
    cs_9126 = res["case_studies"]["9126"]
    assert cs_9126["targets"]["fischer_target"]["sched_d"] == cs_9126["targets"]["fischer_target"]["sched_u"]
    assert cs_9126["targets"]["mal_target"]["sched_d"] != cs_9126["targets"]["mal_target"]["sched_u"]

def test_generated_tables_match_report():
    from exploratory_trace_analysis.generate_report_tables import verify_tables_in_report
    verify_tables_in_report()
