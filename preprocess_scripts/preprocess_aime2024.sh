#!/bin/bash
# Preprocess raw HuggingFace aime-2024 dataset into slime-compatible jsonl.
#
# Raw HF aime-2024 has columns: ID, Problem, Solution, Answer
# Slime expects columns: text, label (, metadata)
#
# Usage: bash preprocess_scripts/preprocess_aime2024.sh
#
# Prerequisites:
#   - datasets/aime-2024/ downloaded via huggingface (contains aime_2024_problems.parquet)
#   - pandas, pyarrow installed

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$WORKSPACE_DIR/datasets/aime-2024"

echo "[$(date)] Preprocessing aime-2024 dataset..."

python3 -c "
import pandas as pd
import json
import os

data_dir = '$DATA_DIR'
in_path = os.path.join(data_dir, 'aime_2024_problems.parquet')
df = pd.read_parquet(in_path)

out_path = os.path.join(data_dir, 'aime-2024.jsonl')
with open(out_path, 'w') as f:
    for _, row in df.iterrows():
        obj = {
            'text': row['Problem'],
            'label': str(row['Answer']),
            'metadata': {'id': row['ID'], 'solution': row['Solution']}
        }
        f.write(json.dumps(obj) + '\n')

print(f'  {len(df)} rows → {out_path}')
"

echo "[$(date)] Done."
