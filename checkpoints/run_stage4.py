"""Stage 4: the full run - live, dream (or no_dream control), silent mid-window change, return.

Preregistered Protocols & Validity Rules:
1. Condition integrity:
   - Sample the learned dream FIRST (recombine=True from experienced regimes).
   - Fischer outer target := realized dream meanings.
   - Mal outer target := controlled mismatch (swapping two uncollected objects with differing values in realized dream).
   - Validity Rule 1: Realized dream must differ from live regime at suspension; otherwise excluded and reported as excluded.
   - Validity Rule 2: Mal mismatch must differ on at least one reachable, uncollected object in suspended state.
2. Return-phase protocol:
   - Suspend mid-episode at fixed step t_suspend=5 (structurally guaranteeing at least 2 objects uncollected).
   - Return resumes that exact suspended state (position, inventory).
   - Episode boundary markers logged in JSON (viewer-only, never in agent's observation stream).
3. No-dream paired control:
   - 'no_dream' condition skips dream phase entirely (zero transitions, zero learning, zero memory writes).
   - Independent RNG streams: live_rng, dream_rng, return_rng, transform_rng.
   - Within-target comparisons reported.
4. Prespecified seeds:
   - Evaluated across 5 declared seeds [13, 23, 42, 77, 99], reporting every result including nulls and exclusions.
"""
import copy, json, random, resource, sys, time
from pathlib import Path
sys.path.insert(0, '.')
from remnant.world import default_world
from remnant.agent import QAgent
from remnant.memory import EpisodicMemory
from remnant.model import WorldModel
from remnant.transform import transform, learned_transform

CONFIG = {
    'prespecified_seeds': [13, 23, 42, 77, 99],
    'live_eps': 300,
    'suspend_at_step': 5,
    'dream_steps': 60,
    'return_steps': 200,
    'max_steps_per_ep': 60,
    'regimes': {
        0: {0: 1.0, 1: -1.0, 2: 0.5, 3: 0.5},
        1: {0: -1.0, 1: 1.0, 2: 0.5, 3: -0.5}
    },
    'validity_rules': {
        'rule1_fischer_differs_from_live': 'Realized dream meanings must differ from the live regime at suspension. Runs where dream == live_regime are excluded.',
        'rule2_mal_differs_on_uncollected': 'Mal controlled mismatch must differ from dream on at least one uncollected object in the suspended state.'
    }
}

class LearnedDreamEnv:
    """Environment backed purely by the learned WorldModel and transformed meanings."""
    def __init__(self, model: WorldModel, transformed_meanings: dict, rng: random.Random):
        self.model = model
        self.meanings = dict(transformed_meanings)
        self.rng = rng
        self.s = None
        self.step_cost = self.model.infer_step_cost()
        self.obj_locs = self.model.infer_object_locations()
        self.pos_to_obj = {pos: idx for idx, pos in self.obj_locs.items()}

    def reset(self, s):
        self.s = s
        return self.s

    def step(self, a):
        s2, _ = self.model.sample_next(self.s, a, self.rng)
        r = self.step_cost
        curr_pos, curr_collected = self.s
        next_pos, next_collected = s2

        if next_pos in self.pos_to_obj:
            obj_idx = self.pos_to_obj[next_pos]
            if obj_idx not in curr_collected:
                r += self.meanings[obj_idx]
                next_collected = curr_collected | {obj_idx}
                s2 = (next_pos, next_collected)

        self.s = s2
        return s2, r

def construct_mal_meanings(realized_dream: dict, uncollected_objs: list):
    """Construct controlled mismatch by swapping two uncollected objects with differing values."""
    mal = dict(realized_dream)
    for i in range(len(uncollected_objs)):
        for j in range(i + 1, len(uncollected_objs)):
            o1, o2 = uncollected_objs[i], uncollected_objs[j]
            if abs(mal[o1] - mal[o2]) > 1e-4:
                mal[o1], mal[o2] = mal[o2], mal[o1]
                return mal, (o1, o2), abs(mal[o1] - mal[o2])
    # Fallback to any two objects if uncollected objects happen to share values
    for o1 in realized_dream:
        for o2 in realized_dream:
            if o1 != o2 and abs(mal[o1] - mal[o2]) > 1e-4:
                mal[o1], mal[o2] = mal[o2], mal[o1]
                return mal, (o1, o2), abs(mal[o1] - mal[o2])
    return mal, None, 0.0

def run_seed(seed):
    # Isolated stream RNGs
    live_rng = random.Random(seed * 1000 + 1)
    dream_rng = random.Random(seed * 1000 + 2)
    return_seed_base = seed * 1000 + 3
    transform_rng = random.Random(seed * 1000 + 4)

    agent = QAgent(memory=EpisodicMemory())
    world = default_world()
    model = WorldModel()

    regime_0 = CONFIG['regimes'][0]
    regime_1 = CONFIG['regimes'][1]

    # --- 1. LIVE PHASE (300 episodes) ---
    # Ep 0..149: regime 0. Ep 150..299: regime 1.
    # Ep 299 suspends mid-episode at suspend_at_step.
    live_log = []
    for ep in range(CONFIG['live_eps'] - 1):
        regime = 0 if ep < 150 else 1
        world.change_meanings(regime_0 if regime == 0 else regime_1)
        s = world.reset((0, 0))
        for t in range(CONFIG['max_steps_per_ep']):
            a = agent.act(s, live_rng)
            s2, r = world.step(a)
            live_log.append(('live', s, a, r, s2))
            agent.memory.append(s, a, r, t)
            agent.learn(s, a, r, s2)
            model.observe(s, a, r, s2, regime=regime)
            s = s2

    # Suspended episode (ep 299)
    current_live_regime = regime_1
    world.change_meanings(current_live_regime)
    s = world.reset((0, 0))
    for t in range(CONFIG['suspend_at_step']):
        a = agent.act(s, live_rng)
        s2, r = world.step(a)
        live_log.append(('live', s, a, r, s2))
        agent.memory.append(s, a, r, t)
        agent.learn(s, a, r, s2)
        model.observe(s, a, r, s2, regime=1)
        s = s2

    suspended_state = world.state()
    uncollected_objs = [idx for idx in [0, 1, 2, 3] if idx not in suspended_state[1]]
    assert len(uncollected_objs) >= 2, f"Validity failure: fewer than 2 objects uncollected at step {CONFIG['suspend_at_step']}"

    # Sample learned dream FIRST
    experienced_regimes = model.get_experienced_regime_meanings()
    realized_dream = learned_transform(experienced_regimes, transform_rng, recombine=True)

    # Validity Rule 1 check: realized dream must differ from live regime at suspension
    diff_from_live = (realized_dream != current_live_regime)
    if not diff_from_live:
        return {
            'seed': seed,
            'status': 'EXCLUDED',
            'exclusion_reason': 'Validity Rule 1 violated: realized dream equals live regime at suspension.',
            'realized_dream': realized_dream,
            'live_regime_at_suspension': current_live_regime,
            'suspended_state': [list(suspended_state[0]), sorted(list(suspended_state[1]))],
            'uncollected_objs': uncollected_objs
        }

    # Construct Fischer and Mal outer targets
    fischer_target = dict(realized_dream)
    mal_target, swapped_pair, swap_diff = construct_mal_meanings(realized_dream, uncollected_objs)

    # Validity Rule 2 check: Mal must differ on at least one uncollected object
    assert swapped_pair is not None and swap_diff > 0, "Validity Rule 2 violated: Mal mismatch had zero difference."
    assert any(o in uncollected_objs for o in swapped_pair), "Validity Rule 2 violated: mismatch not on uncollected object."

    match_vector_fischer = {k: (realized_dream[k] == fischer_target[k]) for k in realized_dream}
    match_vector_mal = {k: (realized_dream[k] == mal_target[k]) for k in realized_dream}

    # Run arms:
    # Under fischer_target: dream vs no_dream
    # Under mal_target: dream vs no_dream
    arms = {}

    for target_name, outer_target, match_vec in [
        ('fischer_target', fischer_target, match_vector_fischer),
        ('mal_target', mal_target, match_vector_mal)
    ]:
        arms[target_name] = {}
        for cond in ['dream', 'no_dream']:
            agent_branch = copy.deepcopy(agent)
            world_branch = default_world()
            world_branch.change_meanings(current_live_regime)
            world_branch.pos = suspended_state[0]
            world_branch.collected = frozenset(suspended_state[1])

            dream_transitions_count = 0
            mem_len_before_dream = len(agent_branch.memory.entries)
            q_table_before_dream = dict(agent_branch.q)

            # DREAM WINDOW (60 steps)
            if cond == 'dream':
                d_rng = random.Random(seed * 1000 + 2)
                dream_env = LearnedDreamEnv(model, realized_dream, d_rng)
                s_d = dream_env.reset(suspended_state)
                mid_step = CONFIG['dream_steps'] // 2
                for dt in range(CONFIG['dream_steps']):
                    if dt == mid_step:
                        world_branch.change_meanings(outer_target)
                    a_d = agent_branch.act(s_d, d_rng)
                    s2_d, r_d = dream_env.step(a_d)
                    agent_branch.memory.append(s_d, a_d, r_d, dt)
                    agent_branch.learn(s_d, a_d, r_d, s2_d)
                    s_d = s2_d
                    dream_transitions_count += 1
            else: # no_dream
                # Silent mid-window change occurs without dream
                world_branch.change_meanings(outer_target)

            # Verification for no_dream control
            if cond == 'no_dream':
                assert dream_transitions_count == 0
                assert len(agent_branch.memory.entries) == mem_len_before_dream
                assert agent_branch.q == q_table_before_dream

            # RETURN PHASE (200 steps)
            # Both conditions use identical return_rng stream
            ret_rng = random.Random(return_seed_base)
            s_ret = suspended_state
            returns = 0.0
            traj = []
            episode_boundaries = []
            step_in_ep = CONFIG['suspend_at_step']

            for rt in range(CONFIG['return_steps']):
                a_ret = agent_branch.act(s_ret, ret_rng)
                s2_ret, r_ret = world_branch.step(a_ret)
                agent_branch.memory.append(s_ret, a_ret, r_ret, rt)
                agent_branch.learn(s_ret, a_ret, r_ret, s2_ret)
                returns += r_ret
                traj.append(round(returns, 3))
                s_ret = s2_ret
                step_in_ep += 1

                # If current episode horizon is reached, reset episode
                if step_in_ep >= CONFIG['max_steps_per_ep']:
                    episode_boundaries.append(rt)
                    s_ret = world_branch.reset((0, 0))
                    step_in_ep = 0

            # Observation stream check
            for entry in agent_branch.memory.entries:
                assert len(entry) == 4, "Observation stream / memory clean check failed"

            arms[target_name][cond] = {
                'return_cumreward': round(returns, 3),
                'dream_transitions_count': dream_transitions_count,
                'episode_boundaries': episode_boundaries,
                'final_trajectory_sample': traj[-5:]
            }

        # Compute within-target contrast: dream return - no_dream return
        dream_r = arms[target_name]['dream']['return_cumreward']
        no_dream_r = arms[target_name]['no_dream']['return_cumreward']
        arms[target_name]['within_target_contrast_dream_minus_nodream'] = round(dream_r - no_dream_r, 3)

    return {
        'seed': seed,
        'status': 'VALID',
        'suspended_pos': list(suspended_state[0]),
        'suspended_collected': sorted(list(suspended_state[1])),
        'uncollected_objs': uncollected_objs,
        'realized_dream': realized_dream,
        'live_regime_at_suspension': current_live_regime,
        'fischer_target': fischer_target,
        'mal_target': mal_target,
        'mal_swapped_pair': list(swapped_pair),
        'mal_swap_diff': round(swap_diff, 3),
        'match_vector_fischer': match_vector_fischer,
        'match_vector_mal': match_vector_mal,
        'arms': arms
    }

def main():
    t0 = time.time()
    results = []
    print(f"Running Remnant Stage 4 across prespecified seeds: {CONFIG['prespecified_seeds']}")
    for seed in CONFIG['prespecified_seeds']:
        res = run_seed(seed)
        results.append(res)

    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    if sys.platform != 'darwin':
        max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    report = {
        'config': CONFIG,
        'wall_seconds': round(time.time() - t0, 2),
        'max_rss_mb': round(max_rss_mb, 2),
        'runs': results
    }

    out_path = Path('checkpoints/stage4_runs.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    # Assert at least one valid run and report all
    valid_runs = [r for r in results if r['status'] == 'VALID']
    assert len(valid_runs) > 0, "Gate Failure: No valid runs generated across prespecified seeds."

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"GATE ASSERTION FAILURE: {e}", file=sys.stderr)
        sys.exit(1)
