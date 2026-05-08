"""
Generate paper figures from real evaluation data.

Fig 2: Grouped bar chart — merge success & crash rate across methods/mixes
Fig 3: BC validation loss curves for all four traffic mixes
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

os.makedirs('figures', exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_BC       = '#7f7f7f'
C_BASELINE = '#ff7f0e'
C_MPC      = '#1f77b4'
C_PPO      = '#2ca02c'

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Method comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════════

# default_mix (from final_evaluation_summary.csv)
default_summary = {
    'BC':       {'success': 0.00, 'crash': 0.10, 'speed': 2.93},
    'Baseline': {'success': 0.90, 'crash': 0.02, 'speed': 6.56},
    'MPC':      {'success': 0.92, 'crash': 0.02, 'speed': 7.89},
    'PPO':      {'success': 0.96, 'crash': 0.00, 'speed': 19.56},
}

# PPO across mixes (from final_eval_*.csv + summary)
ppo_mixes = {
    'all\\_normal':       {'success': 0.84, 'crash': 0.00, 'speed': 19.31},
    'default\\_mix':      {'success': 0.96, 'crash': 0.00, 'speed': 19.56},
    'cautious\\_heavy':   {'success': 0.90, 'crash': 0.00, 'speed': 19.27},
    'aggressive\\_heavy': {'success': 0.96, 'crash': 0.00, 'speed': 19.28},
}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

# ── panel A: method comparison on default_mix ─────────────────────────────────
ax = axes[0]
methods  = list(default_summary.keys())
colors   = [C_BC, C_BASELINE, C_MPC, C_PPO]
successes = [100 * default_summary[m]['success'] for m in methods]
crashes   = [100 * default_summary[m]['crash']   for m in methods]

x     = np.arange(len(methods))
width = 0.35
bars_s = ax.bar(x - width/2, successes, width, label='Merge success',
                color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
bars_c = ax.bar(x + width/2, crashes, width, label='Crash rate',
                color=colors, alpha=0.4, edgecolor='white', linewidth=0.8,
                hatch='//')

# value labels
for bar in bars_s:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
            f'{h:.0f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
for bar in bars_c:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=7, color='#555')

ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 115)
ax.set_ylabel('Rate (%)')
ax.set_title('(a) Method comparison — default\_mix', pad=6)
ax.spines[['top', 'right']].set_visible(False)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#aaaaaa', alpha=0.85, label='Merge success'),
    Patch(facecolor='#aaaaaa', alpha=0.4, hatch='//', label='Crash rate'),
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.7)

# ── panel B: PPO robustness across mixes ─────────────────────────────────────
ax = axes[1]
mixes_lbl = [r'all\_normal', r'default\_mix', r'cautious\_heavy', r'aggressive\_heavy']
ppo_s_vals = [100 * ppo_mixes[m]['success'] for m in ppo_mixes]
ppo_c_vals = [100 * ppo_mixes[m]['crash']   for m in ppo_mixes]

x2    = np.arange(len(mixes_lbl))
bars2 = ax.bar(x2, ppo_s_vals, 0.55, color=C_PPO, alpha=0.85,
               edgecolor='white', linewidth=0.8, label='Merge success')
ax.bar(x2, ppo_c_vals, 0.55, color=C_PPO, alpha=0.35,
       edgecolor='white', linewidth=0.8, hatch='//', label='Crash rate')

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
            f'{h:.0f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(x2)
ax.set_xticklabels([r'all\_norm.', r'default', r'caut.', r'agg.'])
ax.set_ylim(0, 115)
ax.set_ylabel('Rate (%)')
ax.set_title('(b) PPO robustness — trained on default\_mix only', pad=6)
ax.spines[['top', 'right']].set_visible(False)
ax.text(0, 104, '* OOD', fontsize=7, color='#555',
        transform=ax.get_xaxis_transform())
for xi in [0, 2, 3]:
    ax.annotate('*', xy=(xi, ppo_s_vals[xi] + 4), ha='center', fontsize=8, color='#555')

legend_elements2 = [
    Patch(facecolor=C_PPO, alpha=0.85, label='Merge success'),
    Patch(facecolor=C_PPO, alpha=0.35, hatch='//', label='Crash rate (0%)'),
]
ax.legend(handles=legend_elements2, loc='lower right', framealpha=0.7)

plt.tight_layout()
fig.savefig('figures/results_bars.pdf', bbox_inches='tight')
fig.savefig('figures/results_bars.png', bbox_inches='tight', dpi=180)
print("Saved figures/results_bars.pdf and .png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — BC validation loss curves
# ═══════════════════════════════════════════════════════════════════════════════

mix_colors = {
    'default_mix':      C_PPO,
    'all_normal':       C_BASELINE,
    'cautious_heavy':   C_MPC,
    'aggressive_heavy': '#d62728',
}
mix_labels = {
    'default_mix':      r'default\_mix (→ PPO warm-start)',
    'all_normal':       r'all\_normal',
    'cautious_heavy':   r'cautious\_heavy',
    'aggressive_heavy': r'aggressive\_heavy',
}

fig, ax = plt.subplots(figsize=(6, 3.4))

for mix, color in mix_colors.items():
    csv_path = f'plots/bc_policy_{mix}_loss.csv'
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    ax.plot(df['epoch'], df['val_loss'], color=color, lw=1.8,
            label=mix_labels[mix])
    ax.plot(df['epoch'], df['train_loss'], color=color, lw=1.0,
            ls='--', alpha=0.45)

ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('BC training and validation loss (solid = val, dashed = train)', pad=6)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper right', framealpha=0.8)
ax.set_xlim(1, 100)

plt.tight_layout()
fig.savefig('figures/bc_loss.pdf', bbox_inches='tight')
fig.savefig('figures/bc_loss.png', bbox_inches='tight', dpi=180)
print("Saved figures/bc_loss.pdf and .png")
plt.close()
