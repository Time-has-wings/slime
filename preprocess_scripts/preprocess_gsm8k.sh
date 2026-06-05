#!/bin/bash
# Preprocess raw HuggingFace gsm8k dataset into slime-compatible format.
#
# Raw HF gsm8k has columns: question, answer
# Slime expects columns: messages, label
#
# Usage: bash preprocess_scripts/preprocess_gsm8k.sh
#
# Prerequisites:
#   - datasets/gsm8k/ downloaded via download_scripts/download_gsm8k.sh
#   - pandas, pyarrow installed

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$WORKSPACE_DIR/datasets/gsm8k"

echo "[$(date)] Preprocessing gsm8k dataset..."

python3 -c "
import pandas as pd
import os
import re

data_dir = '$DATA_DIR'
for split in ['train', 'test']:
    in_path = os.path.join(data_dir, 'main', f'{split}-00000-of-00001.parquet')
    df = pd.read_parquet(in_path)

    # messages: question as plain string (slime will wrap it as chat message when --apply-chat-template is used)
    df['messages'] = df['question']

    # label: extract final answer after #### (e.g. '#### 72' → '72')
    def extract_final_answer(text):
        m = re.search(r'####\s*(.+)$', str(text))
        return m.group(1).strip() if m else text

    df['label'] = df['answer'].apply(extract_final_answer)

    out_path = os.path.join(data_dir, f'{split}.parquet')
    df.to_parquet(out_path)
    print(f'  {split}: {len(df)} rows → {out_path}')
"

echo "[$(date)] Done."
