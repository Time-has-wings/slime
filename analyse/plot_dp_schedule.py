import matplotlib.pyplot as plt
import numpy as np

# rollout_id -> (rank0, rank1, rank2, rank3) token counts
data = [
    (0,  20.8249, 215906, 215905, 215904, 215904, 863619),
    (1,  20.8856, 210180, 210174, 210154, 210044, 840552),
    (2,  20.9314, 214422, 214418, 214406, 214104, 857350),
    (3,  20.9780, 209727, 209726, 209726, 209536, 838715),
    (4,  21.0239, 241306, 241272, 241246, 241242, 965066),
    (5,  21.0692, 195951, 195950, 195949, 195947, 783797),
    (6,  21.1129, 213664, 213664, 213664, 213660, 854652),
    (7,  21.1607, 207718, 207716, 207716, 207715, 830865),
    (8,  21.2067, 241369, 241369, 241368, 241367, 965473),
    (9,  21.2531, 201973, 201972, 201972, 201972, 807889),
    (10, 21.3322, 227695, 227234, 227030, 227030, 908989),
    (11, 21.3803, 226651, 226651, 226650, 226650, 906602),
    (12, 21.4286, 195933, 195933, 195902, 194866, 782634),
    (13, 21.4722, 212644, 212643, 212643, 212642, 850572),
    (14, 21.5191, 237353, 237353, 237353, 237352, 949411),
    (15, 21.5648, 191199, 191194, 189656, 189585, 761634),
    (16, 21.6044, 182715, 182713, 182713, 182712, 730853),
    (17, 21.6489, 206906, 206904, 206903, 206902, 827615),
    (18, 21.6954, 232221, 232221, 232221, 232220, 928883),
    (19, 21.7444, 205303, 205293, 204968, 204547, 820111),
]

rollout_ids = [d[0] for d in data]
timestamps = [d[1] for d in data]
ranks = np.array([[d[2], d[3], d[4], d[5]] for d in data])
totals = [d[6] for d in data]
percents = ranks / np.array(totals)[:, None] * 100

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- Plot 1: Stacked bar chart of token counts per rank ---
ax1 = axes[0, 0]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
bottom = np.zeros(len(rollout_ids))
for r in range(4):
    ax1.bar(rollout_ids, ranks[:, r], bottom=bottom, color=colors[r],
            label=f'Rank {r}', edgecolor='white', linewidth=0.5)
    bottom += ranks[:, r]
ax1.set_xlabel('Rollout ID')
ax1.set_ylabel('Total Tokens')
ax1.set_title('Token Distribution Across DP Ranks per Rollout')
ax1.legend(loc='upper right')
ax1.set_xticks(rollout_ids)

# --- Plot 2: Percentage deviation from 25% per rank ---
ax2 = axes[0, 1]
for r in range(4):
    deviation = percents[:, r] - 25.0
    ax2.plot(rollout_ids, deviation, 'o-', color=colors[r],
             label=f'Rank {r}', markersize=6, linewidth=1.5)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax2.set_xlabel('Rollout ID')
ax2.set_ylabel('Deviation from 25% (pct points)')
ax2.set_title('Per-Rank Load Balance Deviation')
ax2.legend(loc='upper right')
ax2.set_xticks(rollout_ids)

# --- Plot 3: Total tokens over time (trend) ---
ax3 = axes[1, 0]
ax3.bar(rollout_ids, totals, color='#9C27B0', alpha=0.7, edgecolor='white')
ax3.axhline(y=np.mean(totals), color='red', linestyle='--',
            label=f'Mean: {np.mean(totals):.0f}')
ax3.set_xlabel('Rollout ID')
ax3.set_ylabel('Total Tokens')
ax3.set_title('Total Tokens per Rollout (all ranks combined)')
ax3.legend()
ax3.set_xticks(rollout_ids)

# --- Plot 4: Per-rank summary statistics ---
ax4 = axes[1, 1]
rank_summaries = []
for r in range(4):
    rank_summaries.append({
        'mean': np.mean(ranks[:, r]),
        'std': np.std(ranks[:, r]),
        'min': np.min(ranks[:, r]),
        'max': np.max(ranks[:, r]),
    })
x = np.arange(4)
width = 0.35
means = [r['mean'] for r in rank_summaries]
stds = [r['std'] for r in rank_summaries]
bars = ax4.bar(x, means, width, yerr=stds, color=colors, capsize=8,
               edgecolor='white', linewidth=0.5)
ax4.set_xticks(x)
ax4.set_xticklabels([f'Rank {i}' for i in range(4)])
ax4.set_ylabel('Mean Tokens ± Std')
ax4.set_title('Per-Rank Token Distribution Summary (20 rollouts)')
for i, (bar, mn, sd) in enumerate(zip(bars, means, stds)):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sd + 500,
             f'{mn:.0f}±{sd:.0f}', ha='center', va='bottom', fontsize=9)

fig.suptitle(f'DP Schedule Analysis — Qwen3-4B GRPO\n'
             f'128 samples × 4 DP ranks, dynamic batch, balance_data=True\n'
             f'Total tokens: {sum(totals):,}  |  Mean/rollout: {np.mean(totals):.0f}  |  Range: {min(totals):,}–{max(totals):,}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/linguangming/slime-workspace/slime/dp_schedule_analysis.png', dpi=150, bbox_inches='tight')
print(f"Figure saved to dp_schedule_analysis.png")
print(f"\nSummary stats:")
print(f"{'Rank':<10} {'Mean Tokens':>14} {'Std':>10} {'Min':>12} {'Max':>12} {'CV%':>8}")
for r in range(4):
    cv = np.std(ranks[:, r]) / np.mean(ranks[:, r]) * 100
    print(f"Rank {r}    {np.mean(ranks[:, r]):>12.0f}  {np.std(ranks[:, r]):>8.0f}  "
          f"{np.min(ranks[:, r]):>10}  {np.max(ranks[:, r]):>10}  {cv:>6.1f}%")

# Also save CSV
csv_path = '/tmp/linguangming/slime-workspace/slime/dp_schedule_data.csv'
with open(csv_path, 'w') as f:
    f.write('rollout_id,timestamp_hour,rank0,rank1,rank2,rank3,total\n')
    for d in data:
        f.write(f'{d[0]},{d[1]:.4f},{d[2]},{d[3]},{d[4]},{d[5]},{d[6]}\n')
print(f"CSV saved to dp_schedule_data.csv")
