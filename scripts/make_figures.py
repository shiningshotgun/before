"""Generate publication-ready figures from Remnant recorded run artifacts.
Reads only checkpoints/stage4_runs_v3.json and checkpoints/stage4_runs_v4.json.
Outputs PNGs to docs/images/:
- returns_by_arm.png
- trace_measures.png
- seed99_collections.png
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def make_returns_by_arm(v3_path, out_path):
    with open(v3_path) as f:
        v3 = json.load(f)

    valid_runs = [r for r in v3['runs'] if r['status'] == 'VALID']
    # Seeds 42, 77, 99
    seeds = [r['seed'] for r in valid_runs]
    
    # We will plot 6 conditions: (Seed 42 Fischer, Mal), (Seed 77 Fischer, Mal), (Seed 99 Fischer, Mal)
    labels = []
    dream_vals = []
    untrans_vals = []
    nodream_vals = []

    for r in valid_runs:
        s = r['seed']
        for tgt_key, tgt_name in [('fischer_target', 'Fischer'), ('mal_target', 'Mal')]:
            labels.append(f"Seed {s}\n{tgt_name}")
            arms = r['arms'][tgt_key]
            dream_vals.append(arms['dream']['return_cumreward'])
            untrans_vals.append(arms['dream_untransformed']['return_cumreward'])
            nodream_vals.append(arms['no_dream']['return_cumreward'])

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(x - width, dream_vals, width, label='Dream (Transformed)', color='#1f77b4', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x, untrans_vals, width, label='Dream (Untransformed)', color='#2ca02c', edgecolor='black', linewidth=0.5)
    rects3 = ax.bar(x + width, nodream_vals, width, label='No Dream (Control)', color='#7f7f7f', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Cumulative Return (200 return steps)', fontsize=11, fontweight='medium')
    ax.set_title('Cumulative Return by Arm across Valid Seeds & Target Conditions\n(Excluded under Rule 1: Seeds 13 and 23 where realized dream = live regime)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(frameon=True, fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    # Value labels on top/bottom of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            va = 'bottom' if height >= 0 else 'top'
            y_pos = height + (0.15 if height >= 0 else -0.35)
            ax.annotate(f'{height:+.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                        xytext=(0, 0), textcoords="offset points",
                        ha='center', va=va, fontsize=8, fontweight='semibold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Set appropriate y-limits
    ax.set_ylim(-10.5, 9.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")

def make_trace_measures(v4_path, out_path):
    with open(v4_path) as f:
        v4 = json.load(f)

    valid_runs = [r for r in v4['runs'] if r['status'] == 'VALID']
    # Seeds 42, 77, 99
    seeds = [str(r['seed']) for r in valid_runs]

    # Pre-compute metrics from v4 traces:
    # 1. Post-dream Q-table L1 distance: sum |Q_dream - Q_untrans|
    # 2. Sum |delta Q| for dream and dream_untransformed
    l1_dists = []
    delta_q_dream = []
    delta_q_untrans = []

    for r in valid_runs:
        # Note: dream phase is identical across Fischer and Mal targets because dream env depends on realized dream / suspension regime
        arm_d = r['arms']['fischer_target']['dream']
        arm_u = r['arms']['fischer_target']['dream_untransformed']

        q_d = {((tuple(x['pos']), frozenset(x['col'])), x['action']): x['val'] for x in arm_d['q_table_after_dream']}
        q_u = {((tuple(x['pos']), frozenset(x['col'])), x['action']): x['val'] for x in arm_u['q_table_after_dream']}
        all_keys = set(q_d.keys()) | set(q_u.keys())
        l1 = sum(abs(q_d.get(k, 0.0) - q_u.get(k, 0.0)) for k in all_keys)
        l1_dists.append(round(l1, 4))

        sum_d = sum(abs(u['delta']) for u in arm_d['q_updates_dream'])
        sum_u = sum(abs(u['delta']) for u in arm_u['q_updates_dream'])
        delta_q_dream.append(round(sum_d, 4))
        delta_q_untrans.append(round(sum_u, 4))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    x = np.arange(len(seeds))
    width = 0.35

    # Panel 1: Post-dream Q-table L1 distance
    rects1 = ax1.bar(x, l1_dists, width=0.45, color='#4575b4', edgecolor='black', linewidth=0.6)
    ax1.set_ylabel('L1 Distance $\\sum |Q_{\\mathrm{dream}} - Q_{\\mathrm{untrans}}|$', fontsize=10, fontweight='medium')
    ax1.set_title('Post-Dream Q-Table Divergence (L1)', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=10)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.set_ylim(0, 0.75)
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.4f}',
                     xy=(rect.get_x() + rect.get_width() / 2, h + 0.02),
                     ha='center', va='bottom', fontsize=9, fontweight='semibold')

    # Panel 2: Dream-phase Sum |Delta Q|
    rects2a = ax2.bar(x - width/2, delta_q_dream, width, label='Dream (Transformed)', color='#d73027', edgecolor='black', linewidth=0.6)
    rects2b = ax2.bar(x + width/2, delta_q_untrans, width, label='Dream (Untransformed)', color='#fdae61', edgecolor='black', linewidth=0.6)
    ax2.set_ylabel('Dream Update Magnitude $\\sum |\\Delta Q|$', fontsize=10, fontweight='medium')
    ax2.set_title('Dream-Phase Total Learning Magnitude', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=10)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.set_ylim(0, 0.95)

    for rect in rects2a:
        h = rect.get_height()
        ax2.annotate(f'{h:.4f}',
                     xy=(rect.get_x() + rect.get_width() / 2, h + 0.02),
                     ha='center', va='bottom', fontsize=8, fontweight='semibold')
    for rect in rects2b:
        h = rect.get_height()
        ax2.annotate(f'{h:.4f}',
                     xy=(rect.get_x() + rect.get_width() / 2, h + 0.02),
                     ha='center', va='bottom', fontsize=8, fontweight='semibold')

    fig.suptitle('Stage 4 Internal Trace Divergence during 60-Step Dream Window', fontsize=12, fontweight='bold', y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")

def make_seed99_collections(v4_path, out_path):
    with open(v4_path) as f:
        v4 = json.load(f)

    run99 = [r for r in v4['runs'] if r['seed'] == 99][0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), dpi=300, sharex=True)

    colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a', 3: '#984ea3'}
    obj_names = {0: 'Obj 0', 1: 'Obj 1', 2: 'Obj 2', 3: 'Obj 3'}

    for ax, tgt_key, tgt_title in [(ax1, 'fischer_target', 'Fischer Target Meanings {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.5}'),
                                   (ax2, 'mal_target', 'Mal Target Meanings {0: 0.5, 1: 1.0, 2: 1.0, 3: 0.5}')]:
        arms = run99['arms'][tgt_key]
        y_pos = {'dream': 2, 'dream_untransformed': 1, 'no_dream': 0}

        for cond, y in y_pos.items():
            evs = []
            for e in arms[cond]['return_trace']:
                new = set(e['next_state'][1]) - set(e['state'][1])
                if new:
                    obj = sorted(new)[0]
                    evs.append((e['step'], obj, e['reward']))

            # Plot baseline line
            ax.axhline(y, color='#cccccc', linestyle='-', linewidth=1, zorder=1)

            # Plot events
            for step, obj, r in evs:
                ax.scatter(step, y, s=80, color=colors[obj], edgecolor='black', linewidth=0.8, zorder=3)
                ax.annotate(f"{obj}", (step, y + 0.14), fontsize=8, ha='center', va='bottom', fontweight='bold', color=colors[obj])

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['No Dream', 'Dream Untransformed', 'Dream Transformed'], fontsize=9, fontweight='medium')
        ax.set_ylim(-0.5, 2.7)
        ax.set_title(tgt_title, fontsize=10, fontweight='bold', pad=8)
        ax.grid(axis='x', linestyle=':', alpha=0.6)

    # Custom legend for objects placed at upper center above the top plot
    markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markeredgecolor='black', markersize=8) for i in range(4)]
    ax1.legend(markers, [f"Object {i}" for i in range(4)], loc='upper right', ncol=4, fontsize=9, frameon=True)

    ax2.set_xlabel('Return Phase Step (0 to 199)', fontsize=10, fontweight='medium')
    ax2.set_xlim(-5, 205)

    fig.suptitle('Seed 99: Return-Phase Item Collection Timelines by Arm', fontsize=12, fontweight='bold', y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")

def main():
    v3_path = Path('checkpoints/stage4_runs_v3.json')
    v4_path = Path('checkpoints/stage4_runs_v4.json')
    img_dir = Path('docs/images')
    img_dir.mkdir(parents=True, exist_ok=True)

    make_returns_by_arm(v3_path, img_dir / 'returns_by_arm.png')
    make_trace_measures(v4_path, img_dir / 'trace_measures.png')
    make_seed99_collections(v4_path, img_dir / 'seed99_collections.png')

if __name__ == '__main__':
    main()
