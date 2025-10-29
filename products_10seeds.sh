#!/usr/bin/env bash

# ===== 固定公共超参 =====
DATASET="ogbn-products"
DROPOUT=0.3
LR=0.003
N_PARTS=5
EPOCHS=500
MODEL="graphsage"
SAMPLE_RATE=0.1
LAYERS=3
HIDDEN=128
LOG_EVERY=10
EXTRA_FLAGS=(--use-pp)

# ===== 种子列表 =====
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# ===== 日志目录 =====
OUTDIR="results_multi_seed/products_graphsage"
mkdir -p "${OUTDIR}"

# ===== 1) 第一个 seed：执行分区 =====
S0=${SEEDS[0]}
CUSTOM_PORT=18300  # 临时指定端口
echo "[Run] seed=${S0} (do partition)"
python main.py \
  --dataset "${DATASET}" \
  --dropout "${DROPOUT}" \
  --lr "${LR}" \
  --n-partitions "${N_PARTS}" \
  --n-epochs "${EPOCHS}" \
  --model "${MODEL}" \
  --sampling-rate "${SAMPLE_RATE}" \
  --n-layers "${LAYERS}" \
  --n-hidden "${HIDDEN}" \
  --log-every "${LOG_EVERY}" \
  "${EXTRA_FLAGS[@]}" \
  --fix-seed --seed "${S0}" \
  --port "${CUSTOM_PORT}" --master_addr 127.0.0.1 \
  > "${OUTDIR}/seed_${S0}.log" 2>&1

# ===== 2) 后续种子：跳过分区 =====
for s in "${SEEDS[@]:1}"; do
  echo "[Run] seed=${s} (skip partition)"
  python main.py \
    --dataset "${DATASET}" \
    --dropout "${DROPOUT}" \
    --lr "${LR}" \
    --n-partitions "${N_PARTS}" \
    --n-epochs "${EPOCHS}" \
    --model "${MODEL}" \
    --sampling-rate "${SAMPLE_RATE}" \
    --n-layers "${LAYERS}" \
    --n-hidden "${HIDDEN}" \
    --log-every "${LOG_EVERY}" \
    "${EXTRA_FLAGS[@]}" \
    --skip-partition \
    --fix-seed --seed "${s}" \
    --port "${CUSTOM_PORT}" --master_addr 127.0.0.1 \
    > "${OUTDIR}/seed_${s}.log" 2>&1
done

# ===== 3) 汇总 Test 精度均值 ± 标准差 =====
python - <<'PY'
import os, re, glob, statistics

log_dir = "results_multi_seed/products_graphsage"
p_test = re.compile(r"Test\s+Result\s*\|\s*Accuracy\s+([0-9.]+)%")
p_val  = re.compile(r"Max\s+Validation\s+Accuracy\s+([0-9.]+)%")

test_vals, val_vals = [], []
for f in sorted(glob.glob(os.path.join(log_dir, "seed_*.log"))):
    test_v = None
    val_v = None
    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            mt = p_test.search(line)
            if mt: test_v = float(mt.group(1))
            mv = p_val.search(line)
            if mv: val_v = float(mv.group(1))
    print(os.path.basename(f), "->",
          ("Test %.4f%%" % test_v if test_v is not None else "Test NA"),
          ("| ValMax %.4f%%" % val_v if val_v is not None else ""))
    if test_v is not None: test_vals.append(test_v)
    if val_v  is not None: val_vals.append(val_v)

def show(prefix, arr):
    if not arr: return
    mean = statistics.mean(arr)
    std  = statistics.stdev(arr) if len(arr) >= 2 else 0.0
    print(f"{prefix}: {mean:.2f}% ± {std:.2f}%  (n={len(arr)})")

print("\n=== Aggregation ===")
show("Test Accuracy", test_vals)
show("Max Val Accuracy", val_vals)
PY
