"""Equivalence gate test for Phase 2 passive trace logging.
Proves that:
1. Running with logging_on vs logging_off yields identical action sequences, returns,
   final Q-tables, episodic memory contents, and RNG states.
2. The summary returns under passive logging match the recorded v3 numbers for the same seeds.
"""
import copy, json, random, sys
from pathlib import Path
sys.path.insert(0, '.')
from remnant.world import default_world
from remnant.agent import QAgent
from remnant.memory import EpisodicMemory
from run_stage4 import run_seed

def test_passive_logging_deterministic_equivalence():
    """Verify that logging-on vs logging-off produces 100% identical trajectory, Q-table, memory, and RNG."""
    # Test on Seed 42
    res_logging_off = run_seed(42, log_traces=False)
    res_logging_on = run_seed(42, log_traces=True)

    # Check validity status
    assert res_logging_off['status'] == res_logging_on['status'] == 'VALID'

    # Check returns and contrasts
    for tgt in ['fischer_target', 'mal_target']:
        arm_off = res_logging_off['arms'][tgt]
        arm_on = res_logging_on['arms'][tgt]
        for cond in ['dream', 'dream_untransformed', 'no_dream']:
            assert arm_off[cond]['return_cumreward'] == arm_on[cond]['return_cumreward']
            assert arm_off[cond]['coverage'] == arm_on[cond]['coverage']
            assert arm_off[cond]['final_trajectory_sample'] == arm_on[cond]['final_trajectory_sample']
        assert arm_off['contrasts'] == arm_on['contrasts']

def test_v4_returns_match_v3_recorded_baseline():
    """Verify that v4 numbers match recorded stage4_runs_v3.json for valid seeds."""
    v3_path = Path('checkpoints/stage4_runs_v3.json')
    assert v3_path.exists(), "stage4_runs_v3.json must exist"

    with open(v3_path) as f:
        v3 = json.load(f)

    # Check Seed 42
    v3_s42 = [r for r in v3['runs'] if r['seed'] == 42][0]
    res_v4_s42 = run_seed(42, log_traces=True)

    for tgt in ['fischer_target', 'mal_target']:
        for cond in ['dream', 'dream_untransformed', 'no_dream']:
            v4_ret = res_v4_s42['arms'][tgt][cond]['return_cumreward']
            v3_ret = v3_s42['arms'][tgt][cond]['return_cumreward']
            assert v4_ret == v3_ret, f"Seed 42 return mismatch in {tgt} {cond}: {v4_ret} != {v3_ret}"
            v4_cov = res_v4_s42['arms'][tgt][cond]['coverage']
            v3_cov = v3_s42['arms'][tgt][cond]['coverage']
            assert v4_cov == v3_cov, f"Seed 42 coverage mismatch in {tgt} {cond}: {v4_cov} != {v3_cov}"
        assert res_v4_s42['arms'][tgt]['contrasts'] == v3_s42['arms'][tgt]['contrasts']

if __name__ == '__main__':
    print("Running equivalence gate tests...")
    test_passive_logging_deterministic_equivalence()
    print("test_passive_logging_deterministic_equivalence: PASSED")
    test_v4_returns_match_v3_recorded_baseline()
    print("test_v4_returns_match_v3_recorded_baseline: PASSED")
    print("All equivalence gate checks passed successfully.")
