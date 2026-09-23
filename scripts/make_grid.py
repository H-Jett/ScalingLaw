#!/usr/bin/env python3
"""生成 Chinchilla isoFLOP 实验的模型目录 + 训练配置。

对每个 (模型尺寸, FLOP 预算) 组合, token 数由 D = C/(6N) 反推, 生成一份
LLaMA-Factory 的 pt 训练 YAML。同时输出实验清单 reports/grid.json。

关键约束(论文 App. B): cosine 周期长度必须精确等于训练步数; 因此
learning_rate 调度用 cosine + warmup, 由 max_steps 决定周期。

用法:
    python make_grid.py --root .. --data-dir /path/to/data
    python make_grid.py --root .. --data-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys


# 候选尺寸集: Chinchilla Table A9 的行 (规律 ffw=4d, n_heads*head_dim=d_model)。
# 名称用 HF 口径的实际参数量, 不是 A9 表里那一列(两者口径不同: A9 的 74M 行按 HF 算实为 113.6M)。
MASTER = [
    {"name": "n114M", "hidden": 640, "layers": 10, "heads": 10},
    {"name": "n205M", "hidden": 768, "layers": 18, "heads": 12},
    {"name": "n355M", "hidden": 1024, "layers": 20, "heads": 16},
    {"name": "n542M", "hidden": 1280, "layers": 21, "heads": 10},
    {"name": "n1067M", "hidden": 1792, "layers": 23, "heads": 14},
    {"name": "n1470M", "hidden": 2176, "layers": 22, "heads": 17},
    {"name": "n2303M", "hidden": 2560, "layers": 26, "heads": 20},
]

VOCAB = 50304
SEQ_LEN = 2048


def _params(hidden: int, layers: int) -> float:
    """HF GPTNeoX 参数量: 12*L*d^2 + 2*vocab*d (embedding 不共享) + 13*L*d。"""
    return 12 * layers * hidden**2 + 2 * VOCAB * hidden + 13 * layers * hidden


for _m in MASTER:
    _m["params"] = _params(_m["hidden"], _m["layers"])

# isoFLOP 预算。用 20 tok/param 锚点即 C = 120*N^2, 故 N_opt = sqrt(C/120)。
BUDGETS = [1.0e19, 3.0e19, 6.0e19, 1.2e20]
LADDER_SIZE = 5


def n_opt(c: float) -> float:
    """compute-optimal 参数量 (20 tok/param 锚点)。"""
    return (c / 120.0) ** 0.5


def pick_ladder(c: float, k: int = LADDER_SIZE) -> list[dict]:
    """为给定预算挑 k 个尺寸: 在 log 空间里离 N_opt 最近的 k 个。

    必须跨过 N_opt 两侧, 否则抛物线拟合没有顶点。
    """
    target = n_opt(c)
    ordered = sorted(MASTER, key=lambda m: abs(math.log(m["params"] / target)))
    chosen = sorted(ordered[:k], key=lambda m: m["params"])
    below = [m for m in chosen if m["params"] < target]
    above = [m for m in chosen if m["params"] > target]
    if not below or not above:
        raise SystemExit(f"C={c:.1e} (N_opt={target/1e6:.0f}M) 的阶梯未跨过最优点: {[m['name'] for m in chosen]}")
    return chosen


def make_yaml(model_dir: str, data_dir: str, out_dir: str, d_tokens: float, seq_len: int,
              batch: int, accum: int, lr: float, warmup_frac: float, log_steps: int) -> tuple[str, int, int]:
    """写一份 pt 训练 YAML, 返回 (路径, max_steps, global_batch_tokens)。"""
    global_batch = batch * accum * seq_len
    max_steps = int(round(d_tokens / global_batch))
    warmup_steps = max(1, int(round(max_steps * warmup_frac)))
    yml = f"""### Chinchilla isoFLOP 实验单元 —— 由 make_grid.py 生成, 勿手改
### 注意: 改动 max_steps 会同时改变 cosine 周期长度, 破坏论文 App. B 的要求
model_name_or_path: {model_dir}
trust_remote_code: true
train_from_scratch: true       # 关键: 从 config 随机初始化, 不加载权重

### method
stage: pt
do_train: true
finetuning_type: full

### dataset
dataset: chinchilla_corpus
dataset_dir: {data_dir}
template: default
cutoff_len: {seq_len}
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {out_dir}/output
overwrite_output_dir: true
report_to: none
logging_steps: {log_steps}
save_strategy: "no"            # 中间 checkpoint 对缩放律无用, 省磁盘
plot_loss: false

### train
per_device_train_batch_size: {batch}
gradient_accumulation_steps: {accum}
learning_rate: {lr}
lr_scheduler_type: cosine      # 周期 = max_steps, 必须与 D 匹配
warmup_steps: {warmup_steps}
max_steps: {max_steps}
bf16: true
gradient_checkpointing: true   # 长序列省显存
ddp_timeout: 180000000
"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "train_config.yaml")
    with open(path, "w") as f:
        f.write(yml)
    return path, max_steps, global_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="项目根目录")
    ap.add_argument("--data-dir", required=True, help="tokenized 语料所在目录")
    ap.add_argument("--tokenizer-from", required=True, help="tokenizer 文件源目录 (config+tokenizer)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=32, help="per-device batch (序列数)")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.0e-4, help="占位; 正式调优前用同一值保证可比")
    ap.add_argument("--warmup-frac", type=float, default=0.01, help="warmup 占比 (论文用 1500 步)")
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="只打印清单, 不生成文件")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    scripts = os.path.join(root, "scripts")
    models = os.path.join(root, "models")
    runs = os.path.join(root, "configs")
    reports = os.path.join(root, "reports")
    os.makedirs(reports, exist_ok=True)

    glob_batch = a.batch * a.accum * a.seq_len
    print(f"=== 每预算选定的尺寸阶梯 (全局 batch = {glob_batch:,} token/步) ===")
    print(f"{'预算 C':>9} {'N_opt':>8} | {'尺寸':>8} {'参数':>9} {'D=C/6N':>11} {'tok/param':>10} {'步数':>9}")
    grid, total_flops = [], 0.0
    ladders = {}
    for C in BUDGETS:
        target = n_opt(C)
        ladder = pick_ladder(C)
        ladders[f"{C:.1e}"] = [m["name"] for m in ladder]
        print(f"{C:>9.1e} {target/1e6:>7.0f}M |")
        for m in ladder:
            N, D = m["params"], C / (6 * m["params"])
            steps = int(round(D / glob_batch))
            print(f"{'':>9} {'':>8} | {m['name']:>8} {N/1e6:>8.1f}M {D:>11.3e} {D/N:>10.2f} {steps:>9}")
            grid.append({
                "budget_flops": C, "model": m["name"], "params": N, "hidden": m["hidden"],
                "layers": m["layers"], "heads": m["heads"],
                "tokens": D, "tokens_per_param": D / N, "steps": steps,
                "global_batch_tokens": glob_batch,
                "run_dir": os.path.relpath(os.path.join(runs, f"C{C:.1e}", m["name"]), root),
            })
            total_flops += 6 * N * D

    need = sorted({u["model"] for u in grid})
    print(f"\n=== 需生成的模型目录 ({len(need)} 个): {', '.join(need)} ===")
    by_name = {m["name"]: m for m in MASTER}
    for name in need:
        m = by_name[name]
        out = os.path.join(models, name)
        if a.dry_run:
            print(f"  [dry-run] {name:<8} d={m['hidden']} L={m['layers']} h={m['heads']} "
                  f"-> {m['params']/1e6:.1f}M")
            continue
        subprocess.run(
            [sys.executable, os.path.join(scripts, "gen_model.py"), "--out", out,
             "--hidden", str(m["hidden"]), "--layers", str(m["layers"]), "--heads", str(m["heads"]),
             "--vocab-size", str(VOCAB), "--seq-len", str(a.seq_len),
             "--tokenizer-from", a.tokenizer_from],
            check=True,
        )

    if a.dry_run:
        print(f"\n[dry-run] 未生成文件。单元数 {len(grid)}, 单遍开销 ≈ {total_flops:.2e} FLOPs")
        return

    for u in grid:
        make_yaml(os.path.join(models, u["model"]), a.data_dir,
                  os.path.join(root, u["run_dir"]), u["tokens"], a.seq_len,
                  a.batch, a.accum, a.lr, a.warmup_frac, a.log_steps)

    manifest = {"budgets": BUDGETS, "ladders": ladders, "units": grid,
                "total_flops_1pass": total_flops,
                "note": "total_flops_1pass 是单遍成本; 加超参候选(约 1.6x)才是实际总开销"}
    with open(os.path.join(reports, "grid.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n单元数 {len(grid)}, 单遍开销 ≈ {total_flops:.2e} FLOPs")
    print(f"清单已写入: {os.path.join(reports, 'grid.json')}")


if __name__ == "__main__":
    main()
