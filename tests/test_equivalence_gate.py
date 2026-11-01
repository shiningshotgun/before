"""Equivalence gate test for passive trace logging.
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
    """Verify that logging-on vs logging-off on seed 42 produces 100% full-state deterministic equivalence.
    Asserts exact equality of:
    - Live phase action sequence
    - Pre-suspension RNG states (live_rng, transform_rng)
    - Per arm and per condition:
      - Complete dream action sequences
      - Complete return action sequences
      - Final Q-tables
      - Episodic memory entry contents
      - Memory internal RNG state
      - Final RNG states of dream_act_rng, dream_env_rng, return_act_rng
      - Summary returns, coverage, and contrasts
    """
    # Test on Seed 42
    res_logging_off = run_seed(42, log_traces=False)
    res_logging_on = run_seed(42, log_traces=True)

    # Check validity status
    assert res_logging_off['status'] == res_logging_on['status'] == 'VALID'

    # Check pre-suspension actions and RNG states
    assert res_logging_off['live_actions'] == res_logging_on['live_actions'], "Mismatch in live_actions"
    assert res_logging_off['final_live_rng_state'] == res_logging_on['final_live_rng_state'], "Mismatch in final_live_rng_state"
    assert res_logging_off['final_transform_rng_state'] == res_logging_on['final_transform_rng_state'], "Mismatch in final_transform_rng_state"

    # Check per-target, per-condition full state
    for tgt in ['fischer_target', 'mal_target']:
        arm_off = res_logging_off['arms'][tgt]
        arm_on = res_logging_on['arms'][tgt]
        for cond in ['dream', 'dream_untransformed', 'no_dream']:
            ao = arm_off[cond]
            an = arm_on[cond]

            # 1. Summary measures
            assert ao['return_cumreward'] == an['return_cumreward'], f"{tgt} {cond} return mismatch"
            assert ao['coverage'] == an['coverage'], f"{tgt} {cond} coverage mismatch"
            assert ao['final_trajectory_sample'] == an['final_trajectory_sample'], f"{tgt} {cond} trajectory sample mismatch"

            # 2. Complete action sequences
            assert ao['dream_actions'] == an['dream_actions'], f"{tgt} {cond} dream_actions mismatch"
            assert ao['return_actions'] == an['return_actions'], f"{tgt} {cond} return_actions mismatch"

            # 3. Final Q-tables
            assert ao['final_q'] == an['final_q'], f"{tgt} {cond} final_q mismatch"

            # 4. Episodic memory entries and internal RNG state
            assert ao['final_memory_entries'] == an['final_memory_entries'], f"{tgt} {cond} final_memory_entries mismatch"
            assert ao['final_memory_rng_state'] == an['final_memory_rng_state'], f"{tgt} {cond} final_memory_rng_state mismatch"

            # 5. Stream final RNG states
            assert ao['final_rng_states'] == an['final_rng_states'], f"{tgt} {cond} final_rng_states mismatch"

        assert arm_off['contrasts'] == arm_on['contrasts'], f"{tgt} contrasts mismatch"

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
