#!/usr/bin/env bash

# 固定公共超参（照你原来的命令）
DATASET="reddit"
DROPOUT=0.5
LR=0.01
N_PARTS=2
EPOCHS=3000
MODEL="graphsage"
SAMPLE_RATE=0.1
LAYERS=4
HIDDEN=256
LOG_EVERY=10

# 这些开关与你的命令保持一致
EXTRA_FLAGS=(--inductive --use-pp)

# 种子列表
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# 日志目录（与结果目录分开，便于汇总）
OUTDIR="results_multi_seed/reddit_graphsage"
mkdir -p "${OUTDIR}"

# 第一次运行：执行分区（不加 --skip-partition），并固定 seed
S0=${SEEDS[0]}
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
  > "${OUTDIR}/seed_${S0}.log" 2>&1

# 后续 9 次：跳过分区，固定不同 seed
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
    > "${OUTDIR}/seed_${s}.log" 2>&1
done

# 汇总：从每个 log 中抓取最终 Test 指标，计算均值±标准差
python - <<'PY'
import os, re, glob, statistics
log_dir = "results_multi_seed/reddit_graphsage"

# 你的日志里有 "Test Result | Accuracy 97.19%" 这种格式：
p_test = re.compile(r"Test\s+Result\s*\|\s*Accuracy\s+([0-9.]+)%")
# 同时也抓一下 "Max Validation Accuracy 96.72%"
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
    # 论文里常见 ± 是样本标准差（N-1），这里用 stdev
    std  = statistics.stdev(arr) if len(arr) >= 2 else 0.0
    print(f"{prefix}: {mean:.2f}% ± {std:.2f}%  (n={len(arr)})")

print("\n=== Aggregation ===")
show("Test Accuracy", test_vals)
show("Max Val Accuracy", val_vals)
PY
