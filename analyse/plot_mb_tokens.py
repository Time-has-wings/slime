import re
import numpy as np
import matplotlib.pyplot as plt

with open('/tmp/linguangming/slime-workspace/slime/run.log', 'r') as f:
    content = f.read()

# Strip ANSI codes
content = re.sub(r'\x1b\[[0-9;]*m', '', content)

# Split by "input:" lines to get rollout blocks
rollout_blocks = re.split(r'\[DP schedule\] input: \d+ samples', content)
rollout_blocks = rollout_blocks[1:]  # drop preamble

print(f"Found {len(rollout_blocks)} rollout blocks")

rollouts = []  # [{rank_mbs: {rank: [(mb_idx, total_tokens)]}}]

for block in rollout_blocks:
    rank_mbs = {0: [], 1: [], 2: [], 3: []}

    # Split by rank header
    rank_sections = re.split(r'\[DP schedule\] rank (\d) micro-batch layout', block)

    for i in range(1, len(rank_sections), 2):
        try:
            rank = int(rank_sections[i])
            section = rank_sections[i + 1]

            # Extract mb lines
            for mb_line in section.strip().split('\n'):
                m = re.search(r'mb\[(\d+)\]: (\d+) samples?, total_tokens=(\d+)', mb_line)
                if m:
                    mb_idx = int(m.group(1))
                    total_tokens = int(m.group(3))
                    rank_mbs[rank].append((mb_idx, total_tokens))
        except:
            continue

    if any(len(rank_mbs[r]) > 0 for r in range(4)):
        rollouts.append({'rank_mbs': rank_mbs})

print(f"Parsed {len(rollouts)} rollouts successfully")

# Flatten and check
all_mb_tokens_flat = []
for rollout in rollouts:
    for r in range(4):
        for _, tokens in rollout['rank_mbs'][r]:
            all_mb_tokens_flat.append(tokens)

print(f"Total MB tokens collected: {len(all_mb_tokens_flat)}")

if len(all_mb_tokens_flat) == 0:
    print("ERROR: No MB data parsed!")
    exit(1)

n_rollouts = len(rollouts)
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# --- Plot 1: Per-mb token boxplot per rollout ---
ax1 = axes[0, 0]
all_mb_tokens_per_rollout = []
for rollout in rollouts:
    rollout_tokens = []
    for r in range(4):
        for _, tokens in rollout['rank_mbs'][r]:
            rollout_tokens.append(tokens)
    all_mb_tokens_per_rollout.append(rollout_tokens)

bp = ax1.boxplot(all_mb_tokens_per_rollout, positions=range(n_rollouts),
                  widths=0.6, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#E3F2FD')
    patch.set_edgecolor('#1976D2')
for median in bp['medians']:
    median.set_color('#C62828')
means = [np.mean(t) for t in all_mb_tokens_per_rollout]
ax1.plot(range(n_rollouts), means, 'o-', color='#FF6F00', markersize=5, linewidth=1.5, label='Mean mb tokens')
ax1.set_xlabel('Rollout ID')
ax1.set_ylabel('Micro-batch Total Tokens')
ax1.set_title(f'Micro-batch Token Distribution per Rollout (all 4 ranks)')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# --- Plot 2: Per-rank mean mb tokens ---
ax2 = axes[0, 1]
for r in range(4):
    rank_means = []
    for rollout in rollouts:
        tokens = [t for _, t in rollout['rank_mbs'][r]]
        if tokens:
            rank_means.append(np.mean(tokens))
        else:
            rank_means.append(0)
    ax2.plot(range(len(rank_means)), rank_means, 'o-', color=colors[r],
             label=f'Rank {r}', markersize=5, linewidth=1.5)
ax2.set_xlabel('Rollout ID')
ax2.set_ylabel('Mean MB Tokens')
ax2.set_title('Per-Rank Mean Micro-batch Token Count per Rollout')
ax2.legend()
ax2.grid(alpha=0.3)

# --- Plot 3: Histogram of all mb tokens ---
ax3 = axes[1, 0]
ax3.hist(all_mb_tokens_flat, bins=50, color='#673AB7', alpha=0.7, edgecolor='white')
ax3.axvline(x=np.mean(all_mb_tokens_flat), color='red', linestyle='--',
            label=f'Mean: {np.mean(all_mb_tokens_flat):.0f}')
max_tokens = 9216
ax3.axvline(x=max_tokens, color='orange', linestyle='--',
            label=f'max_tokens_per_gpu={max_tokens}')
ax3.set_xlabel('Micro-batch Total Tokens')
ax3.set_ylabel('Frequency')
ax3.set_title(f'Distribution of All MB Token Counts\n'
              f'Total: {len(all_mb_tokens_flat)} MBs | '
              f'Mean: {np.mean(all_mb_tokens_flat):.0f} | '
              f'Min: {min(all_mb_tokens_flat)} | Max: {max(all_mb_tokens_flat)} | '
              f'Std: {np.std(all_mb_tokens_flat):.0f}')
ax3.legend()

# --- Plot 4: Per-rank mb token stats ---
ax4 = axes[1, 1]
rank_stats = []
for r in range(4):
    tokens = []
    for rollout in rollouts:
        for _, t in rollout['rank_mbs'][r]:
            tokens.append(t)
    rank_stats.append({'mean': np.mean(tokens), 'std': np.std(tokens),
                       'count': len(tokens), 'tokens': tokens})

x = np.arange(4)
means = [s['mean'] for s in rank_stats]
stds = [s['std'] for s in rank_stats]
bars = ax4.bar(x, means, 0.4, yerr=stds, color=colors, capsize=8, edgecolor='white')
ax4.set_xticks(x)
ax4.set_xticklabels([f'Rank {i}' for i in range(4)])
ax4.set_ylabel('Mean MB Tokens ± Std')
ax4.set_title('Per-Rank Micro-batch Token Distribution')
for i, (bar, mn, sd, cnt) in enumerate(zip(bars, means, stds, [s['count'] for s in rank_stats])):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sd + 50,
             f'{mn:.0f}±{sd:.0f}\n({cnt} MBs)', ha='center', va='bottom', fontsize=9)

# --- Additional plot: mb token by rollout heatmap-like ---
fig2, ax_heat = plt.subplots(figsize=(20, 8))

max_mbs = max(max(len(rollout['rank_mbs'][r]) for r in range(4)) for rollout in rollouts)
heatmap_data = np.full((4 * n_rollouts, max_mbs), np.nan)
ytick_labels = []

for ri, rollout in enumerate(rollouts):
    for r in range(4):
        row = ri * 4 + r
        ytick_labels.append(f'R{r}-{ri}')
        for mb_idx, tokens in rollout['rank_mbs'][r]:
            if mb_idx < max_mbs:
                heatmap_data[row, mb_idx] = tokens

# For each rollout group of 4 ranks, get the max
im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax_heat.set_xlabel('Micro-batch Index')
ax_heat.set_ylabel('Rank-Rollout')
ax_heat.set_title(f'Micro-batch Token Distribution Heatmap\n'
                  f'Rows: Rank×Rollout | Columns: MB index | Color: total_tokens')
plt.colorbar(im, ax=ax_heat, label='Total Tokens')

# Only label every 8 rows (every 2 rollouts)
tick_positions = list(range(0, 4 * n_rollouts, 8))
tick_labels_show = [f'R{r%4}-{ri}' for ri, r in 
                    [(p//4, p%4) for p in tick_positions]]
ax_heat.set_yticks(tick_positions)
ax_heat.set_yticklabels(tick_labels_show, fontsize=7)
ax_heat.axhline(y=max_tokens, color='blue', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()

fig2.savefig('/tmp/linguangming/slime-workspace/slime/mb_token_heatmap.png', dpi=150, bbox_inches='tight')
print("Heatmap saved to mb_token_heatmap.png")

fig.suptitle(f'Micro-batch Token Distribution Analysis\n'
             f'{n_rollouts} rollouts × 4 DP ranks | {len(all_mb_tokens_flat)} total MBs | '
             f'Mean: {np.mean(all_mb_tokens_flat):.0f}±{np.std(all_mb_tokens_flat):.0f} | '
             f'max_tokens_per_gpu={max_tokens}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/linguangming/slime-workspace/slime/mb_token_analysis.png', dpi=150, bbox_inches='tight')
print("Figure saved to mb_token_analysis.png")

# Print stats
print("\n=== Per-Rank MB Token Summary ===")
print(f"{'Rank':<8} {'#MBs':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
for r in range(4):
    t = rank_stats[r]['tokens']
    print(f"Rank {r}   {len(t):>6} {np.mean(t):>8.0f} {np.std(t):>8.0f} "
          f"{np.min(t):>8} {np.max(t):>8} "
          f"{np.percentile(t, 50):>8.0f} {np.percentile(t, 95):>8.0f} "
          f"{np.percentile(t, 99):>8.0f}")

# Per-rollout: total mbs and mean mb
print(f"\n=== Per-Rollout MB Summary ===")
print(f"{'Rollout':<8} {'Tot MBs':>8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
for ri, rollout in enumerate(rollouts):
    t = []
    for r in range(4):
        for _, tok in rollout['rank_mbs'][r]:
            t.append(tok)
    print(f"{ri:<8} {len(t):>8} {np.mean(t):>8.0f} {np.std(t):>8.0f} "
          f"{np.min(t):>8} {np.max(t):>8}")

# Check exceeding max
over = [t for t in all_mb_tokens_flat if t > max_tokens]
print(f"\nMBs exceeding max_tokens_per_gpu ({max_tokens}): {len(over)} / {len(all_mb_tokens_flat)}")
if over:
    for v in sorted(set(over))[:20]:
        print(f"  {v} (count: {over.count(v)})")

