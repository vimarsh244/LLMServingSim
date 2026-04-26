#!/bin/bash
set -e
cd /home/marvell/LLMServingSim

CONFIGS="cluster_config/tiered_kv_npu_cxl_cpu.json"
BASE="output/tiered_kv/phaseA/npu_cxl_cpu"


echo "=== fixed_256 ==="
env/bin/python main.py \
  --cluster-config $CONFIGS \
  --dataset dataset/fixed_in128_out512_req256_rate10.jsonl \
  --output $BASE/fixed_256/result.csv \
  --timeseries-output $BASE/fixed_256/timeseries.csv \
  --block-size 16
echo "=== sharegpt_300 ==="
env/bin/python main.py \
  --cluster-config $CONFIGS \
  --dataset dataset/sharegpt_req300_rate10_llama.jsonl \
  --output $BASE/sharegpt_300/result.csv \
  --timeseries-output $BASE/sharegpt_300/timeseries.csv \
  --block-size 16
echo "=== All done ==="
