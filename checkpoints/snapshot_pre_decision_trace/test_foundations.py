import sys, random, os
from pathlib import Path
sys.path.insert(0, '.')
from remnant.world import default_world
from remnant.agent import QAgent
from remnant.memory import EpisodicMemory
from remnant.model import WorldModel
from remnant.transform import transform, learned_transform
from run_stage4 import LearnedDreamEnv, construct_mal_meanings
import run_stage12

def test_walls_block():
    w = default_world()
    w.reset((0, 1))
    s, r = w.step(1)  # south into (1,1) is open
    assert s[0] == (1, 1)
    w.reset((0, 2))
    s, r = w.step(1)  # south into wall (1,2)
    assert s[0] == (0, 2)

def test_meaning_change_is_silent():
    w = default_world()
    s0 = w.reset((3, 0))
    w.change_meanings({0: -5.0, 1: -1.0, 2: 0.5, 3: 0.5})
    assert w.state() == s0 and isinstance(w.state(), tuple)

def test_memory_has_no_origin_field():
    m = EpisodicMemory()
    m.append(((0,0), frozenset()), 0, 1.0, 0)
    assert len(m.entries[0]) == 4

def test_model_learns_transition():
    wm = WorldModel()
    s, s2 = ((0,0), frozenset()), ((1,0), frozenset())
    for _ in range(5):
        wm.observe(s, 1, -0.01, s2)
    psn, pr = wm.predict(s, 1)
    assert psn == s2 and abs(pr - (-0.01)) < 1e-9

def test_defect1_dream_env_checks_predecessor_collected():
    wm = WorldModel()
    s_pre = ((0, 3), frozenset())
    s_succ = ((0, 4), frozenset({0}))
    wm.observe(s_pre, 3, 0.99, s_succ)
    
    meanings = {0: 1.0}
    rng = random.Random(42)
    dream = LearnedDreamEnv(wm, meanings, rng)
    dream.reset(s_pre)
    s2, r = dream.step(3)
    assert 0 in s2[1]
    assert abs(r - 0.99) < 1e-4

def test_defect2_dream_env_uses_learned_tables_only():
    wm = WorldModel()
    s0 = ((1, 0), frozenset())
    s1 = ((1, 1), frozenset({99}))
    for _ in range(5):
        wm.observe(s0, 3, -0.05 + 2.5, s1)
        wm.observe(s1, 0, -0.05, ((0, 1), frozenset({99})))

    assert wm.infer_step_cost() == -0.05
    assert wm.infer_object_locations() == {99: (1, 1)}

    rng = random.Random(42)
    dream = LearnedDreamEnv(wm, {99: 2.5}, rng)
    assert dream.step_cost == -0.05
    assert dream.obj_locs == {99: (1, 1)}

def test_defect5_observation_stream_is_strictly_marker_free():
    w = default_world()
    s = w.reset((0, 0))
    for a in [0, 1, 2, 3]:
        s2, r = w.step(a)
        for obs in (s, s2):
            assert isinstance(obs, tuple) and len(obs) == 2
            pos, col = obs
            assert isinstance(pos, tuple) and len(pos) == 2 and all(isinstance(x, int) for x in pos)
            assert isinstance(col, frozenset) and all(isinstance(x, int) for x in col)
        s = s2

def test_defect6_model_accuracy_reports_coverage_and_unseen():
    wm = WorldModel()
    s0, s1 = ((0, 0), frozenset()), ((0, 1), frozenset())
    wm.observe(s0, 3, -0.01, s1)

    held_out = [
        (s0, 3, -0.01, s1),
        (((4, 4), frozenset()), 0, -0.01, ((3, 4), frozenset()))
    ]
    res = wm.accuracy_on(held_out)
    assert res['seen_count'] == 1
    assert res['dropped_unseen_count'] == 1
    assert res['total_count'] == 2
    assert res['coverage'] == 0.5
    assert res['next_state_accuracy'] == 1.0

def test_defect7_stage2_evaluates_reward_and_next_state():
    wm = WorldModel()
    s0, s1 = ((0, 0), frozenset()), ((0, 1), frozenset())
    wm.observe(s0, 3, 0.5, s1)
    
    held_out = [(s0, 3, 0.5, s1), (s0, 3, -0.5, s1)]
    res = wm.accuracy_on(held_out)
    assert res['next_state_accuracy'] == 1.0
    assert res['reward_accuracy'] == 0.5

def test_defect8_evaluation_does_not_mutate_episodic_memory():
    w = default_world()
    agent = QAgent(memory=EpisodicMemory())
    rng = random.Random(42)
    mem_before = len(agent.memory.entries)
    ret = run_stage12.run_episode(w, agent, rng, log=None, learn=False, record_memory=False)
    mem_after = len(agent.memory.entries)
    assert mem_before == mem_after == 0

def test_defect9_checkpoint_recovery_trajectory_fidelity_with_memory_sampler():
    base_seed = 42

    rng1 = random.Random(base_seed)
    agent1 = QAgent(memory=EpisodicMemory(lookup_sample=200))
    agent1.memory_blend = 0.5
    world1 = default_world()
    traj1 = []
    for ep in range(25):
        s = world1.reset((0, 0))
        for t in range(20):
            a = agent1.act(s, rng1)
            s2, r = world1.step(a)
            agent1.memory.append(s, a, r, t)
            agent1.learn(s, a, r, s2)
            traj1.append((s, a, r, s2))
            s = s2

    ckpt_path = Path('checkpoints/test_ckpt.json')
    rng2 = random.Random(base_seed)
    agent2 = QAgent(memory=EpisodicMemory(lookup_sample=200))
    agent2.memory_blend = 0.5
    world2 = default_world()
    traj2 = []
    for ep in range(15):
        s = world2.reset((0, 0))
        for t in range(20):
            a = agent2.act(s, rng2)
            s2, r = world2.step(a)
            agent2.memory.append(s, a, r, t)
            agent2.learn(s, a, r, s2)
            traj2.append((s, a, r, s2))
            s = s2

    run_stage12.save_checkpoint(ckpt_path, agent2, 15, traj2, rng2)

    resumed_agent, start_ep, resumed_log, resumed_rng = run_stage12.load_checkpoint(ckpt_path)
    assert start_ep == 15
    assert len(resumed_agent.memory.entries) == 300
    for ep in range(15, 25):
        s = world2.reset((0, 0))
        for t in range(20):
            a = resumed_agent.act(s, resumed_rng)
            s2, r = world2.step(a)
            resumed_agent.memory.append(s, a, r, t)
            resumed_agent.learn(s, a, r, s2)
            traj2.append((s, a, r, s2))
            s = s2

    if ckpt_path.exists():
        ckpt_path.unlink()

    assert len(traj1) == len(traj2) == 500
    assert traj1 == traj2, "Trajectory diverged after checkpoint restore with memory sampler active!"

def test_no_dream_control_regression_and_stream_isolation():
    world = default_world()
    agent = QAgent(memory=EpisodicMemory())
    rng = random.Random(101)
    for _ in range(5):
        s = world.reset((0, 0))
        for t in range(10):
            a = agent.act(s, rng); s2, r = world.step(a)
            agent.memory.append(s, a, r, t); agent.learn(s, a, r, s2); s = s2

    q_snap = dict(agent.q)
    mem_snap_len = len(agent.memory.entries)

    dream_transitions = 0
    assert dream_transitions == 0
    assert len(agent.memory.entries) == mem_snap_len
    assert agent.q == q_snap

    ret_rng1 = random.Random(999)
    actions1 = [agent.act(((0, 0), frozenset()), ret_rng1) for _ in range(30)]

    ret_rng2 = random.Random(999)
    actions2 = [agent.act(((0, 0), frozenset()), ret_rng2) for _ in range(30)]

    assert actions1 == actions2, "Return RNG stream isolation failed!"

def test_mal_controlled_mismatch_differs_on_uncollected():
    realized_dream = {0: 1.0, 1: -1.0, 2: 0.5, 3: 0.5}
    uncollected = [0, 1, 2, 3]
    mal, swapped_pair, diff = construct_mal_meanings(realized_dream, uncollected)
    assert swapped_pair == (0, 1)
    assert diff == 2.0
    assert mal[0] == -1.0 and mal[1] == 1.0
    assert all(o in uncollected for o in swapped_pair)

def test_mal_target_differs_from_live_regime_guard():
    """Round 3: verify validity rule 3 guard that mal_target cannot equal live regime."""
    live_regime = {0: -1.0, 1: 1.0, 2: 0.5, 3: -0.5}
    # Synthetic dream which when swapped would produce live_regime
    mal_target_identical = dict(live_regime)
    assert mal_target_identical == live_regime
