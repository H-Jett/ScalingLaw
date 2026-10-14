#!/usr/bin/env python3
"""读取本地训练日志，生成 loss 曲线和 summary.json。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = logging.getLogger("plot_run")

COLORS = ["#2E5EAA", "#D1495B", "#2A9D8F", "#E9C46A", "#8E7DBE", "#F4A261", "#6A994E", "#BC4B51"]


def find_log(run_dir: Path) -> Path | None:
    for path in (run_dir / "trainer_log.jsonl", run_dir / "out" / "trainer_log.jsonl"):
        if path.is_file():
            return path
    return None


def read_curve(run_dir: Path, tokens_per_step: float | None = None) -> dict:
    """读一个 run 的 loss 曲线。"""
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
                continue
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
    return read_budget(run_dir).get("tokens_per_step")


def read_budget(run_dir: Path) -> dict:
    path = run_dir / "budget.json"
    if path.is_file():
        return json.loads(path.read_text())
    LOG.warning("%s 缺少 budget.json，只画 step 曲线", run_dir)
    return {}


def summarize(curve: dict, run_dir: Path) -> dict:
    """保存 loss 统计与预算，算力口径只从 budget.json 读取。"""
    budget = read_budget(run_dir)
    lrs, losses = curve["lrs"], curve["losses"]
    tail = losses[-max(1, len(losses) // 20) :]
    summary = {
        "name": curve["name"],
        "run_dir": curve["run_dir"],
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "tail_mean_loss": sum(tail) / len(tail),
        "n_points": len(losses),
        "final_lr": lrs[-1] if lrs else None,
        "max_lr": max(lrs) if lrs else None,
        "steps": curve["steps"][-1],
        "tokens_per_step": curve["tokens_per_step"],
        "total_tokens": curve["tokens"][-1] if curve["tokens"] else None,
        "N_total": budget.get("n_total"),
        "N_nonemb": budget.get("n_nonemb"),
        "flops": budget.get("flops"),
        "flops_nonemb": budget.get("flops_nonemb"),
        "token_param_ratio": budget.get("token_param_ratio"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def token_axis(curves: list[dict]) -> tuple[float, str]:
    """按最大 token 数选坐标轴单位, 免得小 run 的轴刻度是 "0.000 0.005"。"""
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
            axes[1].plot(
                [t / tok_div for t in c["tokens"]], c["losses"], color=color, lw=1.4, label=label
            )

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
    parser = argparse.ArgumentParser(description="画 loss 曲线并写出摘要")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out", help="图片路径")
    parser.add_argument("--title")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dirs = [Path(p) for p in args.run_dirs]
    curves = [read_curve(p) for p in dirs]
    output = (
        Path(args.out)
        if args.out
        else (dirs[0] / "loss.png" if len(dirs) == 1 else Path("runs/compare.png"))
    )
    plot(curves, output, args.title)
    if not args.no_summary:
        for directory, curve in zip(dirs, curves):
            summary = summarize(curve, directory)
            LOG.info(
                "%s: 最终 loss=%.4f，尾部均值=%.4f",
                summary["name"],
                summary["last_loss"],
                summary["tail_mean_loss"],
            )


if __name__ == "__main__":
    main()
