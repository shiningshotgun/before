"""Stage 4: the full run - live, dream (transformed, untransformed, or no_dream control), silent mid-window change, return.

Preregistered Protocols & Validity Rules:
1. Condition integrity:
   - Sample the learned dream FIRST (recombine=True from experienced regimes).
   - Fischer outer target := realized dream meanings.
   - Mal outer target := controlled mismatch (swapping two uncollected objects with differing values in realized dream).
   - Validity Rule 1: Realized dream must differ from live regime at suspension; otherwise excluded and reported as excluded.
   - Validity Rule 2: Mal mismatch must differ on at least one reachable, uncollected object in suspended state.
   - Validity Rule 3: Mal outer target must differ from the live regime at suspension; otherwise excluded and reported as excluded.
2. Return-phase protocol:
   - Suspend mid-episode at fixed step t_suspend=5 (structurally guaranteeing at least 2 objects uncollected).
   - Return resumes that exact suspended state (position, inventory).
   - Episode boundary markers logged in JSON (viewer-only, never in agent's observation stream).
3. Three paired arms per outer target:
   - 'dream': transformed dream episode inside the learned model with realized_dream meanings.
   - 'dream_untransformed': faithful-replay dream episode inside the learned model with the live regime in effect at suspension.
   - 'no_dream': paired control skipping the dream phase entirely (zero transitions, zero learning, zero memory writes).
   - Contrasts reported within each outer target:
     - dream vs no_dream (overall dream effect)
     - dream_untransformed vs no_dream (untransformed dream baseline effect)
     - dream vs dream_untransformed (attribution of the transformation itself)
4. Rollout coverage tracking:
   - Per dream rollout, count known_steps ((s, a) in model counts) and unknown_steps, recording coverage = known_steps / total_steps.
5. Paired RNG stream architecture:
   - live_rng: seed * 1000 + 1 (generates identical pre-suspension history for all arms)
   - transform_rng: seed * 1000 + 4 (samples realized dream)
   - dream streams: dream and dream_untransformed are paired on identical initial RNG states:
     - dream_act_rng: seed * 1000 + 2 (agent action randomness during dream)
     - dream_env_rng: seed * 1000 + 5 (model transition sampling randomness during dream)
     Both dream arms instantiate their own distinct random.Random instances from these exact seeds.
   - return_act_rng: seed * 1000 + 3 (agent action randomness during return; identical initial state across all three arms)
6. Prespecified seeds:
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
    'output_file': 'checkpoints/stage4_runs_v4.json',
    'rng_stream_seeds': {
        'live': 'seed * 1000 + 1',
        'dream_act': 'seed * 1000 + 2',
        'return_act': 'seed * 1000 + 3',
        'transform': 'seed * 1000 + 4',
        'dream_env': 'seed * 1000 + 5'
    },
    'regimes': {
        0: {0: 1.0, 1: -1.0, 2: 0.5, 3: 0.5},
        1: {0: -1.0, 1: 1.0, 2: 0.5, 3: -0.5}
    },
    'validity_rules': {
        'rule1_fischer_differs_from_live': 'Realized dream meanings must differ from the live regime at suspension. Runs where dream == live_regime are excluded.',
        'rule2_mal_differs_on_uncollected': 'Mal controlled mismatch must differ from dream on at least one uncollected object in the suspended state.',
        'rule3_mal_differs_from_live': 'Mal outer target must differ from the live regime at suspension. Runs where mal_target == live_regime are excluded.'
    }
}

class LearnedDreamEnv:
    """Environment backed purely by the learned WorldModel and specified meanings."""
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

def serialize_state(s):
    pos, col = s
    return [list(pos), sorted(list(col))]

def serialize_q_dict(q_dict):
    out = []
    for (s, a), v in q_dict.items():
        pos, col = s
        out.append({'pos': list(pos), 'col': sorted(list(col)), 'action': a, 'val': round(v, 4)})
    return out

def serialize_rng_state(rng):
    if rng is None:
        return None
    st = rng.getstate()
    return (st[0], list(st[1]), st[2])

def run_seed(seed, log_traces=False):
    # Isolated stream RNGs for pre-suspension
    live_rng = random.Random(seed * 1000 + 1)
    transform_rng = random.Random(seed * 1000 + 4)
    return_seed_base = seed * 1000 + 3
    dream_act_seed = seed * 1000 + 2
    dream_env_seed = seed * 1000 + 5

    agent = QAgent(memory=EpisodicMemory())
    world = default_world()
    model = WorldModel()

    regime_0 = CONFIG['regimes'][0]
    regime_1 = CONFIG['regimes'][1]

    # --- 1. LIVE PHASE (300 episodes) ---
    # Ep 0..149: regime 0. Ep 150..299: regime 1.
    # Ep 299 suspends mid-episode at suspend_at_step.
    live_log = []
    live_actions = []
    for ep in range(CONFIG['live_eps'] - 1):
        regime = 0 if ep < 150 else 1
        world.change_meanings(regime_0 if regime == 0 else regime_1)
        s = world.reset((0, 0))
        for t in range(CONFIG['max_steps_per_ep']):
            a = agent.act(s, live_rng)
            live_actions.append(a)
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
        live_actions.append(a)
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

    # Validity Rule 3 check: Mal target must differ from the live regime at suspension
    if mal_target == current_live_regime:
        return {
            'seed': seed,
            'status': 'EXCLUDED',
            'exclusion_reason': 'Validity Rule 3 violated: mal_target equals live regime at suspension.',
            'realized_dream': realized_dream,
            'mal_target': mal_target,
            'live_regime_at_suspension': current_live_regime,
            'suspended_state': [list(suspended_state[0]), sorted(list(suspended_state[1]))],
            'uncollected_objs': uncollected_objs
        }

    match_vector_fischer = {k: (realized_dream[k] == fischer_target[k]) for k in realized_dream}
    match_vector_mal = {k: (realized_dream[k] == mal_target[k]) for k in realized_dream}

    # Run 3 paired arms under each outer target:
    # 'dream', 'dream_untransformed', 'no_dream'
    arms = {}

    for target_name, outer_target, match_vec in [
        ('fischer_target', fischer_target, match_vector_fischer),
        ('mal_target', mal_target, match_vector_mal)
    ]:
        arms[target_name] = {}
        for cond in ['dream', 'dream_untransformed', 'no_dream']:
            agent_branch = copy.deepcopy(agent)
            world_branch = default_world()
            world_branch.change_meanings(current_live_regime)
            world_branch.pos = suspended_state[0]
            world_branch.collected = frozenset(suspended_state[1])

            dream_transitions_count = 0
            known_steps = 0
            unknown_steps = 0
            mem_len_before_dream = len(agent_branch.memory.entries)
            q_table_before_dream = dict(agent_branch.q)

            # DREAM WINDOW (60 steps)
            dream_trace = []
            q_updates_dream = []
            mem_lookups_dream = []
            dream_actions = []

            if log_traces:
                def on_mem_dream(s, a, val, indices):
                    mem_lookups_dream.append({
                        'state': serialize_state(s),
                        'action': a,
                        'val': round(val, 4) if val is not None else None,
                        'indices': indices
                    })
                def on_dec_dream(s, chosen, q_raws, m_vals, blended, is_random):
                    lookups = list(mem_lookups_dream)
                    mem_lookups_dream.clear()
                    dream_trace.append({
                        'step': dt,
                        'state': serialize_state(s),
                        'chosen_action': chosen,
                        'is_random': is_random,
                        'q_raws': {a: round(v, 4) for a, v in q_raws.items()} if q_raws else None,
                        'm_vals': {a: (round(v, 4) if v is not None else None) for a, v in m_vals.items()} if m_vals else None,
                        'blended_vals': [round(v, 4) for v in blended] if blended else None,
                        'memory_lookups': lookups
                    })
                def on_learn_dream(s, a, r, s_next, old, target, new_q):
                    q_updates_dream.append({
                        'step': dt,
                        'state': serialize_state(s),
                        'action': a,
                        'reward': r,
                        'next_state': serialize_state(s_next),
                        'old_q': round(old, 4),
                        'target': round(target, 4),
                        'new_q': round(new_q, 4),
                        'delta': round(new_q - old, 4)
                    })

                agent_branch.memory.trace_listener = on_mem_dream
                agent_branch.decision_listener = on_dec_dream
                agent_branch.learn_listener = on_learn_dream

            d_act_rng = None
            d_env_rng = None

            if cond in ['dream', 'dream_untransformed']:
                # Both dream arms receive their OWN random.Random instances initialized
                # from IDENTICAL seeds: dream_act_seed and dream_env_seed
                d_act_rng = random.Random(dream_act_seed)
                d_env_rng = random.Random(dream_env_seed)

                # dream meanings: realized_dream if transformed, current_live_regime if untransformed
                d_meanings = realized_dream if cond == 'dream' else current_live_regime
                dream_env = LearnedDreamEnv(model, d_meanings, d_env_rng)
                s_d = dream_env.reset(suspended_state)
                mid_step = CONFIG['dream_steps'] // 2
                for dt in range(CONFIG['dream_steps']):
                    if dt == mid_step:
                        world_branch.change_meanings(outer_target)
                    a_d = agent_branch.act(s_d, d_act_rng)
                    dream_actions.append(a_d)

                    # Coverage tracking: check if (s_d, a_d) exists in model counts
                    if (s_d, a_d) in model.counts:
                        known_steps += 1
                    else:
                        unknown_steps += 1

                    s2_d, r_d = dream_env.step(a_d)
                    agent_branch.memory.append(s_d, a_d, r_d, dt)
                    agent_branch.learn(s_d, a_d, r_d, s2_d)
                    if log_traces and len(dream_trace) > 0 and dream_trace[-1]['step'] == dt:
                        dream_trace[-1]['reward'] = r_d
                        dream_trace[-1]['next_state'] = serialize_state(s2_d)
                    s_d = s2_d
                    dream_transitions_count += 1
            else: # no_dream
                world_branch.change_meanings(outer_target)

            if log_traces:
                agent_branch.memory.trace_listener = None
                agent_branch.decision_listener = None
                agent_branch.learn_listener = None

            q_table_after_dream = dict(agent_branch.q)

            # Verification for no_dream control
            if cond == 'no_dream':
                assert dream_transitions_count == 0
                assert len(agent_branch.memory.entries) == mem_len_before_dream
                assert agent_branch.q == q_table_before_dream

            coverage = round(known_steps / dream_transitions_count, 3) if dream_transitions_count > 0 else None

            # RETURN PHASE (200 steps)
            # All conditions use identical return_act_rng stream initialized from return_seed_base
            return_trace = []
            mem_lookups_ret = []
            return_actions = []

            if log_traces:
                dream_indices_range = set(range(mem_len_before_dream, len(agent_branch.memory.entries)))
                def on_mem_ret(s, a, val, indices):
                    # Check if dream-phase entries were present in sampled indices
                    dream_indices_sampled = [idx for idx in indices if idx in dream_indices_range]
                    mem_lookups_ret.append({
                        'state': serialize_state(s),
                        'action': a,
                        'val': round(val, 4) if val is not None else None,
                        'pool_size': len(indices),
                        'dream_entries_sampled_count': len(dream_indices_sampled)
                    })
                def on_dec_ret(s, chosen, q_raws, m_vals, blended, is_random):
                    lookups = list(mem_lookups_ret)
                    mem_lookups_ret.clear()
                    return_trace.append({
                        'step': rt,
                        'state': serialize_state(s),
                        'chosen_action': chosen,
                        'is_random': is_random,
                        'q_raws': {a: round(v, 4) for a, v in q_raws.items()} if q_raws else None,
                        'm_vals': {a: (round(v, 4) if v is not None else None) for a, v in m_vals.items()} if m_vals else None,
                        'blended_vals': [round(v, 4) for v in blended] if blended else None,
                        'memory_lookups': lookups
                    })

                agent_branch.memory.trace_listener = on_mem_ret
                agent_branch.decision_listener = on_dec_ret

            ret_act_rng = random.Random(return_seed_base)
            s_ret = suspended_state
            returns = 0.0
            traj = []
            episode_boundaries = []
            step_in_ep = CONFIG['suspend_at_step']

            for rt in range(CONFIG['return_steps']):
                a_ret = agent_branch.act(s_ret, ret_act_rng)
                return_actions.append(a_ret)
                s2_ret, r_ret = world_branch.step(a_ret)
                agent_branch.memory.append(s_ret, a_ret, r_ret, rt)
                agent_branch.learn(s_ret, a_ret, r_ret, s2_ret)
                if log_traces and len(return_trace) > 0 and return_trace[-1]['step'] == rt:
                    return_trace[-1]['reward'] = r_ret
                    return_trace[-1]['next_state'] = serialize_state(s2_ret)
                returns += r_ret
                traj.append(round(returns, 3))
                s_ret = s2_ret
                step_in_ep += 1

                if step_in_ep >= CONFIG['max_steps_per_ep']:
                    episode_boundaries.append(rt)
                    s_ret = world_branch.reset((0, 0))
                    step_in_ep = 0

            if log_traces:
                agent_branch.memory.trace_listener = None
                agent_branch.decision_listener = None

            # Observation stream check
            for entry in agent_branch.memory.entries:
                assert len(entry) == 4, "Observation stream / memory clean check failed"

            arm_record = {
                'return_cumreward': round(returns, 3),
                'dream_transitions_count': dream_transitions_count,
                'known_steps': known_steps,
                'unknown_steps': unknown_steps,
                'coverage': coverage,
                'episode_boundaries': episode_boundaries,
                'final_trajectory_sample': traj[-5:],
                'dream_actions': dream_actions,
                'return_actions': return_actions,
                'final_q': serialize_q_dict(agent_branch.q),
                'final_memory_entries': agent_branch.to_dict()['memory_entries'],
                'final_memory_rng_state': agent_branch.memory.get_rng_state(),
                'final_rng_states': {
                    'dream_act_rng': serialize_rng_state(d_act_rng),
                    'dream_env_rng': serialize_rng_state(d_env_rng),
                    'return_act_rng': serialize_rng_state(ret_act_rng)
                }
            }
            if log_traces:
                arm_record['dream_trace'] = dream_trace
                arm_record['q_updates_dream'] = q_updates_dream
                arm_record['return_trace'] = return_trace
                arm_record['q_table_before_dream'] = serialize_q_dict(q_table_before_dream)
                arm_record['q_table_after_dream'] = serialize_q_dict(q_table_after_dream)

            arms[target_name][cond] = arm_record

        # Within-target contrasts
        ret_dream = arms[target_name]['dream']['return_cumreward']
        ret_dream_untrans = arms[target_name]['dream_untransformed']['return_cumreward']
        ret_nodream = arms[target_name]['no_dream']['return_cumreward']

        arms[target_name]['contrasts'] = {
            'dream_minus_nodream': round(ret_dream - ret_nodream, 3),
            'dream_untransformed_minus_nodream': round(ret_dream_untrans - ret_nodream, 3),
            'dream_minus_dream_untransformed': round(ret_dream - ret_dream_untrans, 3)
        }

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
        'live_actions': live_actions,
        'final_live_rng_state': serialize_rng_state(live_rng),
        'final_transform_rng_state': serialize_rng_state(transform_rng),
        'arms': arms
    }

def main():
    t0 = time.time()
    results = []
    print(f"Running Remnant Stage 4 across prespecified seeds: {CONFIG['prespecified_seeds']}")
    for seed in CONFIG['prespecified_seeds']:
        res = run_seed(seed, log_traces=True)
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

    out_path = Path(CONFIG['output_file'])
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
