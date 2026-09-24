#!/usr/bin/env python3
"""把一次(或多次)训练的 loss 曲线画成图, 并写出 summary.json。

只读 LLaMA-Factory 落在 `<output_dir>/trainer_log.jsonl` 里的逐点记录 —— 那是
框架自己写的本地 loss 存档, 不依赖 swanlab / tensorboard, 所以就算上报没配好
曲线也不会丢。

两种用法:
    # 单个 run
    python scripts/plot_run.py runs/86m-1.5b

    # 多个 run 叠在一起(阶段 3 画 isoFLOP 剖面用)
    python scripts/plot_run.py runs/*-1.5b --out runs/isoflop_1.5b.png

产出:
    <run_dir>/loss.png      loss-vs-step + loss-vs-tokens 双面板
    <run_dir>/summary.json  首/末/最小 loss、步数、token 数、吞吐
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境必须, 否则 import pyplot 会挂
import matplotlib.pyplot as plt  # noqa: E402

LOG = logging.getLogger("plot_run")

# 让图好看一点的固定色序(不用默认循环, 多 run 叠加时颜色稳定)
COLORS = ["#2E5EAA", "#D1495B", "#2A9D8F", "#E9C46A", "#8E7DBE", "#F4A261", "#6A994E", "#BC4B51"]


def find_log(run_dir: Path) -> Path | None:
    """在 run 目录里找 loss 记录。

    ⚠️ **优先读 `trainer_log.jsonl`, 而不是 `trainer_state.json`** —— 实测两者
    的 step/loss 完全相同(504/504 点逐值相等), 但:

      - `trainer_log.jsonl` 每行一条直线记录(键: current_steps/loss/lr/epoch/
        elapsed_time/percentage), 结构扁平、容错好(半行 JSON 直接跳过)
      - `trainer_state.json` 是整个训练状态的大对象, 要先全量解析再下钻
        `log_history`; 键名也不一样(step/learning_rate/grad_norm)

    所以 jsonl 是**唯一必需**的 loss 来源, state 只在需要 `grad_norm` 时才读。
    清理权重时两者都会保留。
    """
    for cand in [run_dir / "trainer_log.jsonl", *sorted(run_dir.glob("*/trainer_log.jsonl"))]:
        if cand.exists():
            return cand
    return None


def read_curve(run_dir: Path, tokens_per_step: float | None = None) -> dict:
    """读一个 run 的 loss 曲线。

    尾部可能有一行只有 current_steps/epoch 没有 loss(收尾记录), 这类点丢掉。
    """
    log = find_log(run_dir)
    if log is None:
        raise FileNotFoundError(f"{run_dir} 下没找到 trainer_log.jsonl")

    steps, losses, lrs, epochs = [], [], [], []
    with log.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # 训练被打断时可能留下半行
            if "loss" not in rec:
                continue
            steps.append(rec.get("current_steps", len(steps)))
            losses.append(float(rec["loss"]))
            lrs.append(rec.get("lr"))
            epochs.append(rec.get("epoch"))

    if not steps:
        raise ValueError(f"{log} 里没有有效的 loss 记录")

    step_size = tokens_per_step or load_tokens_per_step(run_dir)
    return {
        "name": run_dir.name,
        "run_dir": str(run_dir),
        "log": str(log),
        "steps": steps,
        "losses": losses,
        "lrs": [x for x in lrs if x is not None],
        "epochs": epochs,
        "tokens": [s * step_size for s in steps] if step_size else None,
        "tokens_per_step": step_size,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "n_points": len(losses),
    }


def load_tokens_per_step(run_dir: Path) -> float | None:
    """从 run 目录里的 train.yaml / budget.json 读每步消费的 token 数。

    每步 token = seq_len * batch * grad_accum * world_size —— 这个数由
    train_abl.py 算好写进 budget.json, 这里只负责读, 不重算(避免口径漂移)。
    """
    for cand in [run_dir / "train.yaml", *sorted(run_dir.glob("*/train.yaml"))]:
        if not cand.exists():
            continue
        import yaml

        cfg = yaml.safe_load(cand.read_text()) or {}
        budget = cfg.get("_budget") or {}
        if budget.get("tokens_per_step"):
            return float(budget["tokens_per_step"])
    for cand in [run_dir / "budget.json", *sorted(run_dir.glob("*/budget.json"))]:
        if cand.exists():
            b = json.loads(cand.read_text())
            if b.get("tokens_per_step"):
                return float(b["tokens_per_step"])
    LOG.warning("%s: 找不到 tokens_per_step, 只画 loss-vs-step", run_dir)
    return None


def read_budget(run_dir: Path) -> dict:
    """读 run 目录里的 budget.json(参数量与算力口径, 由 train_abl.py 写)。"""
    for cand in [run_dir / "budget.json", *sorted(run_dir.glob("*/budget.json"))]:
        if cand.exists():
            return json.loads(cand.read_text())
    LOG.warning("%s: 找不到 budget.json, 算力口径会缺失", run_dir)
    return {}


def summarize(curve: dict, run_dir: Path) -> dict:
    """写 summary.json —— 阶段 4 拟合缩放律时直接从这一堆 json 里汇总。

    这里**只放可信口径**: `flops` / `flops_nonemb` 都取自 budget.json(由公式算出),
    绝不从 HF 的 `train_results.json.total_flos` 取 —— 后者是
    `6 × num_parameters(exclude_embeddings=True) × D`, 在 tie=False 时只排除了
    一个 embedding, 是个既非 N_total 也非 N_nonemb 的中间量, 且偏差随尺寸变化
    (14m 0.542 → 1.2b 0.920), 拿来做横轴会扭曲指数 a。详见 PLAN §9.4。
    """
    lrs = [x for x in curve["lrs"] if x is not None]
    b = read_budget(run_dir)
    summary = {
        "name": curve["name"],
        "run_dir": curve["run_dir"],
        # ---- 纵轴: loss ----
        "first_loss": curve["first_loss"],
        "last_loss": curve["last_loss"],
        "min_loss": curve["min_loss"],
        "final_lr": lrs[-1] if lrs else None,
        "max_lr": max(lrs) if lrs else None,
        "n_points": curve["n_points"],
        # ---- 横轴: N / D / C (全部来自 budget.json, 自算口径) ----
        "steps": curve["steps"][-1] if curve["steps"] else None,
        "tokens_per_step": curve["tokens_per_step"],
        "total_tokens": (curve["steps"][-1] * curve["tokens_per_step"]) if curve["tokens_per_step"] else None,
        "N_total": b.get("n_total"),
        "N_nonemb": b.get("n_nonemb"),
        "flops": b.get("flops"),  # C = 6·N_total·D      ← 主口径
        "flops_nonemb": b.get("flops_nonemb"),  # C = 6·N_nonemb·D   ← 敏感性分析口径
        "token_param_ratio": b.get("token_param_ratio"),
    }
    # 末尾 loss 的均值比单点更稳(最后几个点还会抖), 拟合时常用这个
    tail = curve["losses"][-max(1, len(curve["losses"]) // 20) :]
    summary["tail_mean_loss"] = sum(tail) / len(tail)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def token_axis(curves: list[dict]) -> tuple[float, str]:
    """按最大 token 数选坐标轴单位, 免得小 run 的轴刻度是 "0.000 0.005"。

    返回 (除数, 单位名)。
    """
    max_tok = max((c["tokens"][-1] for c in curves if c["tokens"]), default=0)
    return (1e9, "B") if max_tok >= 1e9 else (1e6, "M")


def plot(curves: list[dict], out_path: Path, title: str | None = None) -> None:
    """画 loss 曲线。有 token 信息就画双面板(step / tokens), 否则单面板。"""
    has_tokens = all(c["tokens"] for c in curves)
    n_panels = 2 if has_tokens else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False)
    axes = axes[0]

    tok_div, tok_unit = token_axis(curves)

    for i, c in enumerate(curves):
        color = COLORS[i % len(COLORS)]
        label = c["name"]
        axes[0].plot(c["steps"], c["losses"], color=color, lw=1.4, label=label)
        if has_tokens:
            axes[1].plot([t / tok_div for t in c["tokens"]], c["losses"], color=color, lw=1.4, label=label)

    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("loss vs. step")
    if has_tokens:
        axes[1].set_xlabel(f"tokens ({tok_unit})")
        axes[1].set_ylabel("loss")
        axes[1].set_title("loss vs. tokens seen")

    for ax in axes:
        ax.grid(alpha=0.3, ls=":")
        ax.legend(fontsize=9)

    # 各 run 白吃了个数量级差别的步数时, 线性轴会把短 run 压扁 —— 步数跨度
    # 超过 50 倍就切 log 轴
    span = max(c["steps"][-1] for c in curves) / max(1, min(c["steps"][-1] for c in curves))
    if span > 50:
        axes[0].set_xscale("log")

    fig.suptitle(title or "training loss", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    LOG.info("图已保存: %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="画 loss 曲线 + 写 summary.json")
    ap.add_argument("run_dirs", nargs="+", help="run 目录(含 trainer_log.jsonl)")
    ap.add_argument("--out", default=None, help="图片输出路径; 多 run 时默认 runs/compare.png, 单 run 时默认 <run>/loss.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-summary", action="store_true", help="不写 summary.json")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dirs = [Path(d) for d in a.run_dirs]
    curves = [read_curve(d) for d in dirs]

    if a.out:
        out_path = Path(a.out)
    elif len(dirs) == 1:
        out_path = dirs[0] / "loss.png"
    else:
        out_path = Path("runs/compare.png")

    plot(curves, out_path, title=a.title)

    tok_div, tok_unit = token_axis(curves)
    print(f"\n{'run':<24} {'点数':>5} {'首loss':>9} {'末loss':>9} {'最小loss':>9} {'步数':>8} {f'tokens({tok_unit})':>10}")
    for c in curves:
        tok = f"{c['tokens'][-1]/tok_div:.3f}" if c["tokens"] else "-"
        print(
            f"{c['name']:<24} {c['n_points']:>5} {c['first_loss']:>9.4f} {c['last_loss']:>9.4f} "
            f"{c['min_loss']:>9.4f} {c['steps'][-1]:>8} {tok:>10}"
        )

    if not a.no_summary:
        for d, c in zip(dirs, curves):
            s = summarize(c, d)
            LOG.info(
                "%s: 末 loss %.4f (尾部均值 %.4f) | N_total=%s C=%.3e",
                s["name"], s["last_loss"], s["tail_mean_loss"],
                f"{s['N_total']:,}" if s["N_total"] else "-",
                s["flops"] or 0,
            )


if __name__ == "__main__":
    main()
