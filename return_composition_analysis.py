"""Return-phase reward composition analysis for before stage4_runs_v4.json.

Read-only arithmetic over the recorded decision-trace artifact. No RNG, no
simulation, no training. Object identity of each collection comes from the
collected-set delta between the recorded state and next_state of each step.

Usage: python3 return_composition_analysis.py [path-to-stage4_runs_v4.json]
"""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/stage4_runs_v4.json'
dec = json.JSONDecoder()
txt = open(path).read()
i = txt.index('"runs"'); i = txt.index('[', i); pos = i + 1
runs = []
while True:
    while txt[pos] in ' \n\t,': pos += 1
    if txt[pos] == ']': break
    obj, end = dec.raw_decode(txt, pos)
    runs.append(obj)
    pos = end

def st(x):
    return (tuple(x[0]), tuple(x[1]))

def events(arm):
    evs, total, step_cost_total = [], 0.0, 0.0
    for e in arm['return_trace']:
        r = e['reward']; total += r
        new = set(e['next_state'][1]) - set(e['state'][1])
        if new:
            evs.append((e['step'], sorted(new)[0] if len(new) == 1 else sorted(new), round(r, 3)))
        else:
            step_cost_total += r
    return evs, round(total, 3), round(step_cost_total, 3)

for r in runs:
    if r['status'] != 'VALID':
        continue
    print(f"\n##### SEED {r['seed']} #####")
    for tgt in ['fischer_target', 'mal_target']:
        arms = r['arms'][tgt]
        print(f"--- {tgt} ---")
        res = {}
        for cond in ['dream', 'dream_untransformed', 'no_dream']:
            evs, total, scost = events(arms[cond])
            res[cond] = (evs, total, scost)
            print(f"  {cond:20s} total={total:7.3f} (recorded {arms[cond]['return_cumreward']}) "
                  f"stepcost={scost:8.3f} collections={len(evs)}")
            print(f"     events(step,obj,reward): {evs}")
        d, u = res['dream'], res['dream_untransformed']
        same = [(s, o) for s, o, _ in d[0]] == [(s, o) for s, o, _ in u[0]]
        print(f"  dream-vs-untrans collections identical (steps+objects): {same}")
        if not same:
            ds = {(s, o) for s, o, _ in d[0]}; us = {(s, o) for s, o, _ in u[0]}
            print(f"    only in dream: {sorted(ds - us)}")
            print(f"    only in untrans: {sorted(us - ds)}")
            for name, ev in [('dream', d[0]), ('untrans', u[0])]:
                per = {}
                for s, o, rw in ev:
                    per[o] = round(per.get(o, 0) + rw, 3)
                print(f"    per-object collected reward {name}: {per}")
        dt = arms['dream']['return_trace']; ut = arms['dream_untransformed']['return_trace']
        rewards_ident = all(a['reward'] == b['reward'] for a, b in zip(dt, ut))
        print(f"  dream-vs-untrans rewards identical on all 200 steps: {rewards_ident}")
        if not rewards_ident:
            n_diff = sum(1 for a, b in zip(dt, ut) if a['reward'] != b['reward'])
            print(f"    reward-differing steps: {n_diff}")
