import re
import csv
import numpy as np
import matplotlib.pyplot as plt

with open('/tmp/linguangming/slime-workspace/slime/run.log', 'r') as f:
    content = f.read()

content = re.sub(r'\x1b\[[0-9;]*m', '', content)

rollout_blocks = re.split(r'\[DP schedule\] input: \d+ samples', content)
rollout_blocks = rollout_blocks[1:]

rollouts = []
for block in rollout_blocks:
    rank_mbs = {0: [], 1: [], 2: [], 3: []}
    rank_sections = re.split(r'\[DP schedule\] rank (\d) micro-batch layout', block)
    for i in range(1, len(rank_sections), 2):
        try:
            rank = int(rank_sections[i])
            section = rank_sections[i + 1]
            for mb_line in section.strip().split('\n'):
                m = re.search(r'mb\[(\d+)\]: (\d+) samples?, total_tokens=(\d+)', mb_line)
                if m:
                    rank_mbs[rank].append((int(m.group(1)), int(m.group(3))))
        except:
            continue
    if any(len(rank_mbs[r]) > 0 for r in range(4)):
        rollouts.append({'rank_mbs': rank_mbs})

# Save CSV
csv_path = '/tmp/linguangming/slime-workspace/slime/mb_token_data.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['rollout_id', 'rank', 'mb_idx', 'total_tokens'])
    for ri, rollout in enumerate(rollouts):
        for r in range(4):
            for mb_idx, tokens in rollout['rank_mbs'][r]:
                writer.writerow([ri, r, mb_idx, tokens])
print(f"CSV saved: {csv_path} ({sum(1 for _ in open(csv_path)) - 1} rows)")

# Build flat data
all_mb_tokens_flat = []
for rollout in rollouts:
    for r in range(4):
        for _, tokens in rollout['rank_mbs'][r]:
            all_mb_tokens_flat.append(tokens)

n_rollouts = len(rollouts)
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
max_tokens = 9216

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- Plot 1: MB token boxplot per rollout (all ranks) ---
ax1 = axes[0, 0]
all_mb_tokens_per_rollout = []
for rollout in rollouts:
    rollout_tokens = []
    for r in range(4):
        for _, tokens in rollout['rank_mbs'][r]:
            rollout_tokens.append(tokens)
    all_mb_tokens_per_rollout.append(rollout_tokens)

bp = ax1.boxplot(all_mb_tokens_per_rollout, positions=range(n_rollouts),
                  widths=0.6, patch_artist=True, showfliers=True, flierprops={'markersize': 2, 'alpha': 0.5})
for patch in bp['boxes']:
    patch.set_facecolor('#E3F2FD')
    patch.set_edgecolor('#1976D2')
for median in bp['medians']:
    median.set_color('#C62828')
means = [np.mean(t) for t in all_mb_tokens_per_rollout]
ax1.plot(range(n_rollouts), means, 'o-', color='#FF6F00', markersize=5,
         linewidth=1.5, label=f'Mean ({np.mean(all_mb_tokens_flat):.0f})')
ax1.axhline(y=np.mean(all_mb_tokens_flat), color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax1.set_xlabel('Rollout ID')
ax1.set_ylabel('Micro-batch Total Tokens')
ax1.set_title(f'Micro-batch Token Distribution per Rollout (all 4 ranks)\n'
              f'{n_rollouts} rollouts × 4 ranks = {len(all_mb_tokens_flat)} MBs total')
ax1.legend(fontsize=8)

# --- Plot 2: Per-rank mean MB tokens across rollouts ---
ax2 = axes[0, 1]
for r in range(4):
    rank_means = []
    for rollout in rollouts:
        tokens = [t for _, t in rollout['rank_mbs'][r]]
        rank_means.append(np.mean(tokens) if tokens else 0)
    ax2.plot(range(len(rank_means)), rank_means, 'o-', color=colors[r],
             label=f'Rank {r}', markersize=5, linewidth=1.5)
ax2.set_xlabel('Rollout ID')
ax2.set_ylabel('Mean MB Tokens per Rank')
ax2.set_title('Per-Rank Mean Micro-batch Tokens per Rollout')
ax2.legend()
ax2.grid(alpha=0.3)

# --- Plot 3: Histogram of all MB tokens ---
ax3 = axes[1, 0]
ax3.hist(all_mb_tokens_flat, bins=50, color='#673AB7', alpha=0.75, edgecolor='white')
ax3.axvline(x=np.mean(all_mb_tokens_flat), color='red', linestyle='--',
            label=f'Mean: {np.mean(all_mb_tokens_flat):.0f}')
ax3.axvline(x=max_tokens, color='orange', linestyle='--',
            label=f'max_tokens_per_gpu={max_tokens}')
ax3.fill_betweenx([0, 200], max_tokens - 100, max_tokens, color='orange', alpha=0.15)
ax3.set_xlabel('Micro-batch Total Tokens')
ax3.set_ylabel('Frequency')
ax3.set_title(f'Distribution of All {len(all_mb_tokens_flat)} Micro-batch Token Counts\n'
              f'Mean: {np.mean(all_mb_tokens_flat):.0f} | '
              f'Std: {np.std(all_mb_tokens_flat):.0f} | '
              f'Min: {min(all_mb_tokens_flat)} | '
              f'Max: {max(all_mb_tokens_flat)} (cap={max_tokens})')
ax3.legend()

# --- Plot 4: Per-rank MB token distribution boxplot ---
ax4 = axes[1, 1]
rank_data = []
for r in range(4):
    tokens = []
    for rollout in rollouts:
        for _, t in rollout['rank_mbs'][r]:
            tokens.append(t)
    rank_data.append(tokens)
bp4 = ax4.boxplot(rank_data, positions=range(4), widths=0.5, patch_artist=True, showfliers=True,
                   flierprops={'markersize': 2, 'alpha': 0.4})
for i, patch in enumerate(bp4['boxes']):
    patch.set_facecolor(colors[i])
    patch.set_edgecolor('#333333')
for median in bp4['medians']:
    median.set_color('white')
ax4.set_xticks(range(4))
ax4.set_xticklabels([f'Rank {i}' for i in range(4)])
ax4.set_ylabel('Micro-batch Total Tokens')
ax4.set_title(f'Per-Rank Micro-batch Token Distribution\n'
              f'Each rank: {len(rank_data[0])} MBs')
ax4.grid(axis='y', alpha=0.3)
# Add mean text per rank
for r in range(4):
    ax4.annotate(f'μ={np.mean(rank_data[r]):.0f}\nσ={np.std(rank_data[r]):.0f}',
                 xy=(r, np.mean(rank_data[r])), fontsize=8, ha='center', va='bottom',
                 color='red', fontweight='bold')

fig.suptitle(f'Micro-batch Token Distribution\n'
             f'20 rollouts × 4 DP ranks | {len(all_mb_tokens_flat)} total MBs | '
             f'Mean MB: {np.mean(all_mb_tokens_flat):.0f} ± {np.std(all_mb_tokens_flat):.0f} tokens | '
             f'max_tokens_per_gpu: {max_tokens}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/linguangming/slime-workspace/slime/mb_token_analysis_v2.png', dpi=150, bbox_inches='tight')
print("Figure saved: mb_token_analysis_v2.png")

# Print per-rank stats
print("\n=== Per-Rank MB Stats ===")
for r, t in enumerate(rank_data):
    print(f"Rank {r}: {len(t)} MBs | mean={np.mean(t):.0f} | std={np.std(t):.0f} | "
          f"min={np.min(t)} | max={np.max(t)} | p99={np.percentile(t, 99):.0f}")

# Print per-rollout MB count and mean
print(f"\n{'Rollout':<8} {'MBs':>6} {'Mean MB':>9} {'Std':>9} {'Min':>8} {'Max':>8}")
for ri, rollout in enumerate(rollouts):
    t = [tok for r in range(4) for _, tok in rollout['rank_mbs'][r]]
    print(f"{ri:<8} {len(t):>6} {np.mean(t):>9.0f} {np.std(t):>9.0f} {np.min(t):>8} {np.max(t):>8}")

print(f"\nOver limit ({max_tokens}): {sum(1 for t in all_mb_tokens_flat if t > max_tokens)} / {len(all_mb_tokens_flat)}")

