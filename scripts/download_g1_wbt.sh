#!/bin/bash
# 批量下载 Unitree G1 WBT 数据集 (共8个, ~77GB)
# 使用 hf-mirror.com，支持断点续传和超时重试

set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300

BASE_DIR="/disk_n/zzf/flip/data/unitree_G1_WBT"
mkdir -p "$BASE_DIR"

DATASETS=(
  "unitreerobotics/G1_WBT_Inspire_Collect_Clothes_MainCamOnly"
  "unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly"
  "unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly"
  "unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine"
  "unitreerobotics/G1_WBT_Inspire_Put_Clothes_Into_Basket"
  "unitreerobotics/G1_WBT_Brainco_Pickup_Pillow"
  "unitreerobotics/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher"
  "unitreerobotics/G1_WBT_Brainco_Make_The_Bed"
)

MAX_RETRIES=5

for ds in "${DATASETS[@]}"; do
  name="${ds#*/}"
  dest="$BASE_DIR/$name"
  echo "=========================================="
  echo "Downloading: $ds"
  echo "Target: $dest"
  echo "=========================================="

  retry=0
  while [ $retry -lt $MAX_RETRIES ]; do
    if python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$ds',
    repo_type='dataset',
    local_dir='$dest',
)
print('Done: $name')
"; then
      echo "Success: $name"
      break
    else
      retry=$((retry + 1))
      echo "Retry $retry/$MAX_RETRIES for $name ..."
      sleep 5
    fi
  done

  if [ $retry -eq $MAX_RETRIES ]; then
    echo "FAILED after $MAX_RETRIES retries: $name"
  fi

  echo ""
done

echo "All downloads complete!"
du -sh "$BASE_DIR"/*
echo ""
du -sh "$BASE_DIR"
