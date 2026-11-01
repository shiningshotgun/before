"""Descriptive measures analysis for Stage 4 decision traces.
Arithmetic-only: reads stage4_runs_v4.json and computes trace measures without training or RNG.
"""
import json, sys
from pathlib import Path

def analyze_traces(artifact_path='checkpoints/stage4_runs_v4.json'):
    path = Path(artifact_path)
    if not path.exists():
        print(f"Error: artifact not found at {artifact_path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    for run in data['runs']:
        seed = run['seed']
        status = run['status']
        print(f"=== SEED {seed} ({status}) ===")
        if status != 'VALID':
            print(f"Exclusion reason: {run['exclusion_reason']}")
            continue

        for tgt in ['fischer_target', 'mal_target']:
            arm_dream = run['arms'][tgt]['dream']
            arm_untrans = run['arms'][tgt]['dream_untransformed']
            arm_nodream = run['arms'][tgt]['no_dream']

            # Measure 1: Action Divergence during Dream
            d_trace = arm_dream['dream_trace']
            u_trace = arm_untrans['dream_trace']

            first_reward_diff_step = None
            for step in range(len(d_trace)):
                r_d = d_trace[step].get('reward')
                r_u = u_trace[step].get('reward')
                if r_d != r_u:
                    first_reward_diff_step = step
                    break

            action_matches_pre = []
            action_matches_post = []
            for step in range(len(d_trace)):
                a_d = d_trace[step]['chosen_action']
                a_u = u_trace[step]['chosen_action']
                match = (a_d == a_u)
                if first_reward_diff_step is None or step < first_reward_diff_step:
                    action_matches_pre.append(match)
                else:
                    action_matches_post.append(match)

            frac_pre = sum(action_matches_pre) / len(action_matches_pre) if action_matches_pre else 1.0
            frac_post = sum(action_matches_post) / len(action_matches_post) if action_matches_post else 1.0

            # Measure 2: Learning Difference (sum of |delta Q|)
            q_up_d = arm_dream['q_updates_dream']
            q_up_u = arm_untrans['q_updates_dream']

            sum_abs_delta_d = sum(abs(u['delta']) for u in q_up_d)
            sum_abs_delta_u = sum(abs(u['delta']) for u in q_up_u)

            # Post-dream Q-table L1 difference
            q_after_d = {((tuple(x['pos']), frozenset(x['col'])), x['action']): x['val'] for x in arm_dream['q_table_after_dream']}
            q_after_u = {((tuple(x['pos']), frozenset(x['col'])), x['action']): x['val'] for x in arm_untrans['q_table_after_dream']}

            all_keys = set(q_after_d.keys()) | set(q_after_u.keys())
            q_diff_l1 = sum(abs(q_after_d.get(k, 0.0) - q_after_u.get(k, 0.0)) for k in all_keys)

            # Measure 3: Return-Phase Difference
            ret_d = arm_dream['return_trace']
            ret_u = arm_untrans['return_trace']
            ret_no = arm_nodream['return_trace']

            match_ret_d_u = sum(1 for s1, s2 in zip(ret_d, ret_u) if s1['chosen_action'] == s2['chosen_action']) / len(ret_d)
            match_ret_d_no = sum(1 for s1, s2 in zip(ret_d, ret_no) if s1['chosen_action'] == s2['chosen_action']) / len(ret_d)

            div_step_d_u = next((i for i, (s1, s2) in enumerate(zip(ret_d, ret_u)) if s1['chosen_action'] != s2['chosen_action']), None)
            div_step_d_no = next((i for i, (s1, s2) in enumerate(zip(ret_d, ret_no)) if s1['chosen_action'] != s2['chosen_action']), None)

            # State alignment in return
            same_state_count = 0
            matched_act_at_same_state = 0
            for s1, s2 in zip(ret_d, ret_u):
                st1 = (tuple(s1['state'][0]), tuple(s1['state'][1]))
                st2 = (tuple(s2['state'][0]), tuple(s2['state'][1]))
                if st1 == st2:
                    same_state_count += 1
                    if s1['chosen_action'] == s2['chosen_action']:
                        matched_act_at_same_state += 1

            # Measure 4: Memory Usage & Blend Impact
            pool_sizes = []
            dream_counts = []
            m_blend_contributions = []
            q_argmax_changes = 0
            evaluated_steps = 0
            changed_with_dream = 0
            changed_without_dream = 0

            for s in ret_d:
                if not s['is_random'] and s['blended_vals'] is not None and s['q_raws'] is not None:
                    evaluated_steps += 1
                    q_raw_best = max(range(4), key=lambda a: s['q_raws'][str(a)])
                    blend_best = max(range(4), key=lambda a: s['blended_vals'][a])
                    if q_raw_best != blend_best:
                        q_argmax_changes += 1
                        has_dream = any(lk['dream_entries_sampled_count'] > 0 for lk in s['memory_lookups'])
                        if has_dream:
                            changed_with_dream += 1
                        else:
                            changed_without_dream += 1
                    for a in range(4):
                        bv = s['blended_vals'][a]
                        qv = s['q_raws'][str(a)]
                        m_blend_contributions.append(abs(bv - qv))

                for lk in s['memory_lookups']:
                    pool_sizes.append(lk['pool_size'])
                    dream_counts.append(lk['dream_entries_sampled_count'])

            frac_lookups_with_dream = sum(1 for c in dream_counts if c > 0) / len(dream_counts) if dream_counts else 0.0
            avg_blend_impact = sum(m_blend_contributions) / len(m_blend_contributions) if m_blend_contributions else 0.0
            avg_dream_in_pool = sum(dream_counts) / len(dream_counts) if dream_counts else 0.0

            # Argmax agreement on jointly greedy return steps
            same_argmax_count = 0
            eval_greedy_count = 0
            for sd, su in zip(ret_d, ret_u):
                if not sd['is_random'] and not su['is_random'] and sd['blended_vals'] and su['blended_vals']:
                    eval_greedy_count += 1
                    b_d = max(range(4), key=lambda a: sd['blended_vals'][a])
                    b_u = max(range(4), key=lambda a: su['blended_vals'][a])
                    if b_d == b_u:
                        same_argmax_count += 1
            frac_same_argmax = same_argmax_count / eval_greedy_count if eval_greedy_count else 1.0

            print(f"  Target: {tgt}")
            print(f"    Dream divergence: first reward diff step={first_reward_diff_step}")
            print(f"    Dream action match: pre-diff={frac_pre:.3f} ({len(action_matches_pre)} steps), post-diff={frac_post:.3f} ({len(action_matches_post)} steps)")
            print(f"    Q-learning sum |delta Q|: dream={sum_abs_delta_d:.4f}, untrans={sum_abs_delta_u:.4f}, L1 diff in Q-table={q_diff_l1:.4f}")
            print(f"    Return action match: dream vs untrans={match_ret_d_u:.3f} (first div at step {div_step_d_u}), dream vs nodream={match_ret_d_no:.3f} (first div at step {div_step_d_no})")
            print(f"    Return state alignment: {same_state_count}/200 identical steps (match at identical: {matched_act_at_same_state}/{same_state_count})")
            print(f"    Return memory lookups with dream entries: {frac_lookups_with_dream:.3f} ({sum(1 for c in dream_counts if c > 0)}/{len(dream_counts)}), avg dream count: {avg_dream_in_pool:.2f}")
            print(f"    Average memory blend impact |blended - Q|: {avg_blend_impact:.4f}")
            print(f"    Memory altered Q argmax on greedy steps: {q_argmax_changes}/{evaluated_steps} ({q_argmax_changes/evaluated_steps:.3f})")
            print(f"    Argmax shifts with dream entries: {changed_with_dream}, without dream entries: {changed_without_dream}")
            print(f"    Argmax agreement dream vs untrans on greedy steps: {same_argmax_count}/{eval_greedy_count} ({frac_same_argmax:.3f})")

if __name__ == '__main__':
    artifact = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/stage4_runs_v4.json'
    analyze_traces(artifact)
