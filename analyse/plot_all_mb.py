import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Read the data
path = '/tmp/linguangming/slime-workspace/slime/mb_tokens_by_rank.csv'
rollouts = {}
with open(path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rid = int(row['rollout_id'])
        midx = int(row['mb_idx'])
        if rid not in rollouts:
            rollouts[rid] = {0: [], 1: [], 2: [], 3: []}
        for r in range(4):
            val = row[f'rank{r}_tokens']
            if val:
                rollouts[rid][r].append(int(val))

n_rollouts = len(rollouts)
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
max_tokens = 9216

fig, axes = plt.subplots(2, 3, figsize=(22, 12))

# ------ Plot 1: Per-mb tokens as colored dots (all rollouts) ------
ax1 = axes[0, 0]
x_pos = 0
x_ticks, x_labels = [], []
for rid in sorted(rollouts):
    for r in range(4):
        tokens = rollouts[rid][r]
        xs = range(x_pos, x_pos + len(tokens))
        ax1.scatter(xs, tokens, s=3, c=colors[r], alpha=0.6, linewidths=0)
        x_pos += len(tokens)
    x_ticks.append(x_pos - sum(len(rollouts[rid][r]) for r in range(4)) / 2)
    x_labels.append(str(rid))

ax1.axhline(y=max_tokens, color='orange', linestyle='--', linewidth=1, alpha=0.5, label=f'cap={max_tokens}')
ax1.axhline(y=np.mean([t for rid in rollouts for r in range(4) for t in rollouts[rid][r]]),
            color='red', linestyle=':', linewidth=1, alpha=0.5, label=f'mean={np.mean([t for rid in rollouts for r in range(4) for t in rollouts[rid][r]]):.0f}')
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_labels, fontsize=7)
ax1.set_xlabel('Rollout ID')
ax1.set_ylabel('MB Total Tokens')
ax1.set_title(f'All {sum(len(rollouts[rid][r]) for rid in rollouts for r in range(4))} Micro-batch Token Counts\n(4 ranks × 20 rollouts)')
ax1.legend([Patch(color=c) for c in colors] + [plt.Line2D([0],[0],color='orange',ls='--'),
           plt.Line2D([0],[0],color='red',ls=':')],
           [f'Rank {r}' for r in range(4)] + [f'cap={max_tokens}', f'mean={np.mean([t for rid in rollouts for r in range(4) for t in rollouts[rid][r]]):.0f}'],
           fontsize=7, loc='upper right')

# ------ Plot 2: Per-rank boxplot for all rollouts combined ------
ax2 = axes[0, 1]
rank_all = []
for r in range(4):
    all_t = []
    for rid in sorted(rollouts):
        all_t.extend(rollouts[rid][r])
    rank_all.append(all_t)
bp = ax2.boxplot(rank_all, widths=0.5, patch_artist=True, showfliers=True,
                  flierprops={'markersize': 2, 'alpha': 0.3})
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors[i])
    patch.set_edgecolor('#333')
for median in bp['medians']:
    median.set_color('white')
ax2.set_xticklabels([f'Rank {i}\n({len(rank_all[i])} MBs)' for i in range(4)])
ax2.set_ylabel('MB Total Tokens')
ax2.set_title('Per-Rank MB Token Distribution (all 20 rollouts)')
ax2.axhline(y=max_tokens, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax2.grid(axis='y', alpha=0.3)

# ------ Plot 3: Stacked area of token distribution ------
ax3 = axes[0, 2]
bin_edges = np.arange(0, 10000, 200)
for r in range(4):
    all_t = rank_all[r]
    hist, _ = np.histogram(all_t, bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax3.fill_between(centers, hist, alpha=0.4, color=colors[r], label=f'Rank {r}')
ax3.axvline(x=max_tokens, color='orange', linestyle='--', linewidth=1.5, label=f'cap={max_tokens}')
ax3.set_xlabel('MB Total Tokens')
ax3.set_ylabel('Frequency')
ax3.set_title('MB Token Distribution by Rank (overlapping)')
ax3.legend(fontsize=7)

# ------ Plot 4: Per-rollout per-rank MB count vs mean ------
ax4 = axes[1, 0]
for r in range(4):
    xs, ys = [], []
    for rid in sorted(rollouts):
        tokens = rollouts[rid][r]
        if tokens:
            xs.append(np.mean(tokens))
            ys.append(len(tokens))
    ax4.scatter(xs, ys, c=colors[r], s=30, alpha=0.7, label=f'Rank {r}', edgecolors='white', linewidth=0.5)
ax4.set_xlabel('Mean MB Tokens (this rollout, this rank)')
ax4.set_ylabel('Number of MBs')
ax4.set_title('Per-Rank MB Count vs Mean Token Size')
ax4.legend(fontsize=7)
ax4.grid(alpha=0.3)

# ------ Plot 5: Per-rank per-rollout total tokens ------
ax5 = axes[1, 1]
for r in range(4):
    totals = [sum(rollouts[rid][r]) for rid in sorted(rollouts)]
    ax5.plot(sorted(rollouts), totals, 'o-', color=colors[r], markersize=5, linewidth=1.5, label=f'Rank {r}')
ax5.set_xlabel('Rollout ID')
ax5.set_ylabel('Total Tokens')
ax5.set_title('Per-Rank Total Tokens per Rollout')
ax5.legend(fontsize=7)
ax5.grid(alpha=0.3)

# ------ Plot 6: Heatmap-like: per rank per rollout mean mb token ------
ax6 = axes[1, 2]
heatmap = np.zeros((n_rollouts, 4))
for ri, rid in enumerate(sorted(rollouts)):
    for r in range(4):
        tokens = rollouts[rid][r]
        heatmap[ri, r] = np.mean(tokens) if tokens else 0
im = ax6.imshow(heatmap.T, aspect='auto', cmap='YlOrRd', vmin=7000, vmax=8500)
ax6.set_xticks(range(n_rollouts))
ax6.set_xticklabels([str(rid) for rid in sorted(rollouts)], fontsize=7)
ax6.set_yticks(range(4))
ax6.set_yticklabels([f'Rank {r}' for r in range(4)])
ax6.set_title('Mean MB Tokens: Rank × Rollout')
plt.colorbar(im, ax=ax6, label='Mean MB Tokens', shrink=0.8)
# Annotate
for ri in range(n_rollouts):
    for r in range(4):
        val = heatmap[ri, r]
        ax6.text(ri, r, f'{val:.0f}', ha='center', va='center', fontsize=6,
                 color='white' if val > 8100 else 'black')

# Global stats
all_tokens_flat = [t for rid in rollouts for r in range(4) for t in rollouts[rid][r]]

fig.suptitle(f'Micro-batch Token Distribution — 20 Rollouts × 4 DP Ranks\n'
             f'Total MBs: {len(all_tokens_flat)} | '
             f'Global Mean: {np.mean(all_tokens_flat):.0f} ± {np.std(all_tokens_flat):.0f} | '
             f'Cap: {max_tokens} | '
             f'Over cap: {sum(1 for t in all_tokens_flat if t > max_tokens)}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/linguangming/slime-workspace/slime/all_mb_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: all_mb_analysis.png")
