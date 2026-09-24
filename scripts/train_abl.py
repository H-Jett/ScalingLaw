#!/usr/bin/env python3
"""一次消融/缩放实验 = 一条命令: 规划 token 预算 -> 备数据 -> 生成配置 -> 训练 -> 出图。

这个脚本把"跑一个 run"的全部环节串起来, 目的是让阶段 3 的网格能用一条循环跑完,
且每个 run 都自留证据(配置、预算、日志、曲线), 事后不用回忆当时怎么跑的。

一个 run 的产出(runs/<name>/):
    train.yaml      LLaMA-Factory 配置(可直接重跑)
    budget.json     参数口径: N / D / C=6ND / 每步 token / 总步数
    data/           本次用到的语料切片(dataset_info.json + raw/ 软链接)
    out/            框架输出: trainer_log.jsonl(本地 loss 记录)、TB events、trainer_state.json
    train.log       完整 stdout(含框架日志)
    loss.png        训练结束后自动生成的曲线图
    summary.json    首/末/最小 loss、token 数等, 给阶段 4 汇总用

三个关键设计(踩过坑的地方, 别改回去):

1. **cosine 周期必须正好等于 max_steps**。论文 App. B 测过: 学习率退火周期若比
   实际训练步数长 25% 以上, 性能明显变差。所以 max_steps 由 token 预算反推出来,
   scheduler 交给 HF 按 max_steps 自己算, 不手工写步数。

2. **token 预算换算成步数时用的是"全局"batch**, 即
   `tokens_per_step = seq * batch * grad_accum * world_size`。
   漏掉 world_size 会让实际训练的 token 数只有预期的 1/4(D 少 4 倍, 直接毁掉
   整个 isoFLOP 剖面), 这是最容易错的一处。

3. **每步的 loss 一定落盘**(框架自己写 trainer_log.jsonl)。swanlab 只当"好看的上报
   通道", 它挂了不该导致实验记录丢失 —— 所以图是从本地 jsonl 画的, 不是从 swanlab。

用法:
    # 跑一个 run: 86m 模型, 看 1.5B tokens
    python scripts/train_abl.py --model models/configs/pythia-86m --tokens 1.5e9 --name 86m-d1.5b

    # 用 Chinchilla 的 20 tok/param 反推 token 数
    python scripts/train_abl.py --model models/configs/pythia-86m --ratio 20 --name 86m-r20

    # 只打印计划不训练(检查预算算得对不对)
    python scripts/train_abl.py --model models/configs/pythia-184m --tokens 3e9 --name x --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

# 同目录的兄弟模块(直接跑脚本时 sys.path[0] 就是 scripts/, 但被 import 时不一定)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402
import plot_run  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CORPUS_DIR = "/volume/data/hjiang02/open_source/datasets/fineweb/sample/100BT"
LOG_DIR = REPO / "logs"
RUN_ROOT = REPO / "runs"

LOG = logging.getLogger("train_abl")


# --------------------------------------------------------------------------- #
# 参数口径
# --------------------------------------------------------------------------- #
def param_counts(model_dir: Path) -> dict:
    """从 config.json 算参数量。

    公式(main/PLAN.md §2, 已对 pythia 官方 9 档校验, 误差 0):
        N_非emb = 12·L·d² + 13·L·d + 2·d
        N_总   = N_非emb + 2·vocab·d
    训练完还会用框架日志里的实际参数量复核一遍(见 cross_check_params)。
    """
    cfg = json.loads((model_dir / "config.json").read_text())
    d = int(cfg["hidden_size"])
    layers = int(cfg["num_hidden_layers"])
    vocab = int(cfg["vocab_size"])
    nonemb = 12 * layers * d * d + 13 * layers * d + 2 * d
    total = nonemb + 2 * vocab * d
    return {
        "d_model": d,
        "n_layers": layers,
        "n_heads": cfg.get("num_attention_heads"),
        "vocab": vocab,
        "n_nonemb": nonemb,
        "n_embed": 2 * vocab * d,
        "n_total": total,
    }


def tokens_per_step(seq: int, batch: int, grad_accum: int, world_size: int) -> int:
    """全局每步消费的 token 数。**world_size 不能漏**(见模块注释第 2 条)。"""
    return seq * batch * grad_accum * world_size


def plan_budget(
    params: dict,
    seq: int,
    batch: int,
    grad_accum: int,
    world_size: int,
    tokens: float | None,
    ratio: float | None,
) -> dict:
    """把"想训多少 token"翻译成"跑多少步", 并把算力口径一并记下来。"""
    if tokens is None:
        if ratio is None:
            raise ValueError("必须给 --tokens 或 --ratio 之一")
        # Chinchilla 的 20 tok/param 口径是按**总**参数量(含 embedding)算的
        tokens = ratio * params["n_total"]
        basis = f"ratio={ratio} × N_total"
    else:
        basis = f"显式指定 tokens={tokens:.3e}"

    tps = tokens_per_step(seq, batch, grad_accum, world_size)
    steps = max(1, round(tokens / tps))
    return {
        "seq_len": seq,
        "micro_batch": batch,
        "grad_accum": grad_accum,
        "world_size": world_size,
        "tokens_per_step": tps,
        "tokens_requested": tokens,
        "steps": steps,
        "tokens_actual": steps * tps,  # 步数取整后实际会训到的 token 数
        "basis": basis,
        "n_total": params["n_total"],
        "n_nonemb": params["n_nonemb"],
        # C = 6ND 里的 N 用总参数量(含 embedding), 与论文口径一致
        "flops": 6 * params["n_total"] * steps * tps,
        "token_param_ratio": (steps * tps) / params["n_total"],
    }


# --------------------------------------------------------------------------- #
# 配置生成
# --------------------------------------------------------------------------- #
def build_config(
    model_dir: Path,
    data_dir: Path,
    out_dir: Path,
    dataset_name: str,
    budget: dict,
    lr: float,
    warmup_steps: int,
    grad_checkpointing: bool,
    save_final: bool,
    swanlab: dict,
    num_workers: int,
    save_steps: int | None,
) -> dict:
    """生成 LLaMA-Factory 的 train YAML。

    字段刻意只留必需的 —— 每个多余字段都是将来读到这个日志时要重新判断的噪声。
    """
    steps = budget["steps"]
    cfg = {
        "### model": None,
        "model_name_or_path": str(model_dir),
        "trust_remote_code": True,
        "train_from_scratch": True,  # 从 config 随机初始化, 不加载任何权重
        "### method": None,
        "stage": "pt",  # 预训练(纯 LM, 只读正文列)
        "finetuning_type": "full",
        "do_train": True,
        "### data": None,
        "dataset": dataset_name,
        "dataset_dir": str(data_dir),
        "template": "default",  # 其 eos 是 <|endoftext|>, 与 pythia tokenizer 一致
        "cutoff_len": budget["seq_len"],
        "packing": True,  # 把短样本拼满, 否则 2048 的块里大半是 padding
        "preprocessing_num_workers": num_workers,
        "overwrite_cache": True,
        "### output": None,
        "output_dir": str(out_dir),
        "overwrite_output_dir": True,
        "report_to": "tensorboard",  # 本地 loss 的第二份记录(第一份是 trainer_log.jsonl)
        "logging_steps": 1,  # 每步都记 —— 小模型单步很便宜, 曲线密一点更好看拐点
        "save_strategy": "steps" if save_final else "no",
        "plot_loss": False,  # 用我们自己的 plot_run.py, 图更全(带 token 轴)
        "### train": None,
        "per_device_train_batch_size": budget["micro_batch"],
        "gradient_accumulation_steps": budget["grad_accum"],
        "learning_rate": lr,
        "max_steps": steps,  # 由 token 预算反推, cosine 周期与之精确对齐
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "bf16": True,
        "gradient_checkpointing": grad_checkpointing,
        "ddp_timeout": 180000000,
        "### swanlab": None,
    }
    if save_final:
        # 只存最后一个, 且只留 1 份 —— 检查点是磁盘大户
        cfg["save_steps"] = save_steps or steps
        cfg["save_total_limit"] = 1

    cfg.update(swanlab)
    # "### xxx" 的 None 值只是 YAML 里的分节标题(见 dump_yaml), 要保留;
    # 其余 None 是真该丢的
    return {k: v for k, v in cfg.items() if v is not None or k.startswith("### ")}


def dump_yaml(cfg: dict, path: Path) -> None:
    """写 YAML, 保留分节标题。

    yaml.safe_dump 会把带 None 的 key 写成 `key: null`, 这里手工拼:
    标题行输出成 `### 分节名`, 其余交给 safe_dump。
    """
    lines: list[str] = []
    for k, v in cfg.items():
        if k.startswith("### "):
            lines.append(f"\n{k}")
            continue
        lines.append(yaml.safe_dump({k: v}, default_flow_style=False, sort_keys=False, allow_unicode=True).rstrip())
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# swanlab
# --------------------------------------------------------------------------- #
def swanlab_config(args: argparse.Namespace, run_name: str) -> dict:
    """拼 swanlab 配置。cloud 模式需要 API key, 拿不到就退回 local 并告警。"""
    if args.swanlab_mode == "off":
        LOG.info("swanlab: 已关闭")
        return {}

    mode = args.swanlab_mode
    api_key = os.environ.get("SWANLAB_API_KEY")
    if mode == "cloud" and not api_key:
        LOG.warning("swanlab: 要 cloud 模式但环境变量 SWANLAB_API_KEY 没设 -> 退回 local 模式")
        LOG.warning("swanlab: 要用云端上报, 先 `export SWANLAB_API_KEY=<你的key>`")
        mode = "local"

    cfg = {
        "use_swanlab": True,
        "swanlab_project": args.swanlab_project,
        "swanlab_run_name": run_name,
        "swanlab_mode": mode,
    }
    if mode == "local":
        cfg["swanlab_logdir"] = str(RUN_ROOT / "_swanlab")
    elif api_key:
        cfg["swanlab_api_key"] = api_key  # 由环境变量传入, 不写进任何文件

    LOG.info("swanlab: mode=%s project=%s run=%s", mode, args.swanlab_project, run_name)
    return cfg


# --------------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------------- #
# 只把框架自己的日志行转给 logger。**不能简单搜 "error"** —— 数据集预览会把
# 网页正文原样打出来, 里面出现 "Error contacting activation server" 这种句子,
# 曾被误当成错误转出。所以要求行首带框架日志前缀。
_LOG_MARKERS = ("train_loss", "trainable params", "[INFO|", "[WARNING|", "[ERROR|", "Traceback (most recent call last)")


def run_training(yaml_path: Path, log_path: Path) -> int:
    """跑训练, stdout 同时进控制台和日志文件。返回退出码。"""
    env = dict(os.environ, DISABLE_VERSION_CHECK="1")
    proc = subprocess.Popen(
        ["llamafactory-cli", "train", str(yaml_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    with log_path.open("w") as fh:
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            # 框架输出很吵(带进度条), 只把关键行转给 logger, 其余留在文件里
            if any(k in line for k in _LOG_MARKERS):
                LOG.info("  | %s", line.rstrip())
    return proc.wait()


def cross_check_params(run_dir: Path, params: dict) -> None:
    """拿框架日志里的实际参数量复核公式算出来的值。

    公式错了整条缩放律的横轴就错了, 所以每次都对一遍 —— 对不上就在日志里喊出来。
    """
    msg = None
    for cand in [run_dir / "train.log", *sorted(run_dir.glob("*/train.log"))]:
        if cand.exists():
            for line in cand.read_text(errors="ignore").splitlines():
                if "trainable params:" in line and "all params:" in line:
                    msg = line
    if not msg:
        LOG.warning("复核: 框架日志里没找到参数量, 跳过")
        return
    try:
        actual = int(msg.split("trainable params:")[1].split("|")[0].strip().replace(",", ""))
    except (IndexError, ValueError):
        LOG.warning("复核: 参数量那行解析不了: %s", msg)
        return
    if actual == params["n_total"]:
        LOG.info("复核: 实际参数量 %s == 公式 %s ✓", f"{actual:,}", f"{params['n_total']:,}")
    else:
        LOG.error(
            "复核失败: 实际 %s, 公式 %s, 差 %s —— 横轴口径有问题, 别直接拿去做拟合",
            f"{actual:,}", f"{params['n_total']:,}", f"{actual - params['n_total']:,}",
        )


def bj_now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")


def setup_logging(run_name: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    con = logging.StreamHandler()
    con.setFormatter(fmt)
    LOG.addHandler(con)
    fh = logging.FileHandler(LOG_DIR / f"{run_name}.log")
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="跑一个训练 run: 备数据 -> 生成配置 -> 训练 -> 出图")
    ap.add_argument("--model", required=True, help="模型 config 目录, 如 models/configs/pythia-86m")
    ap.add_argument("--name", required=True, help="run 名, 决定 runs/<name>/ 和 swanlab run 名")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tokens", type=float, help="训练 token 数(显式)")
    g.add_argument("--ratio", type=float, help="训练 token 数 = ratio × N_total (Chinchilla 用 20)")
    ap.add_argument("--seq", type=int, default=2048, help="序列长度; 实测 2048 时 MFU 最高(见 PLAN §1.1)")
    ap.add_argument("--batch", type=int, default=8, help="每卡 micro batch")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--world-size", type=int, default=None, help="GPU 数, 默认自动探测")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-steps", type=int, default=None, help="默认 max(20, 2%% of steps)")
    ap.add_argument("--grad-checkpointing", action="store_true", help="大模型放不下时开")
    ap.add_argument("--save-final", action="store_true", help="存最后一个检查点(默认不存, 拟合只需 loss)")
    ap.add_argument("--save-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--corpus", default=CORPUS_DIR)
    ap.add_argument("--run-root", default=str(RUN_ROOT))
    ap.add_argument("--swanlab-mode", default="local", choices=["local", "cloud", "off"])
    ap.add_argument("--swanlab-project", default="scalinglaw")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划, 不训练")
    ap.add_argument("--allow-reuse", action="store_true", help="run 目录已存在时覆盖重跑")
    a = ap.parse_args()

    setup_logging(a.name)
    t0 = time.time()

    model_dir = Path(a.model).resolve()
    if not (model_dir / "config.json").exists():
        LOG.error("找不到 %s/config.json", model_dir)
        return 2

    world_size = a.world_size
    if world_size is None:
        try:
            import torch

            world_size = torch.cuda.device_count() or 1
        except Exception:
            world_size = 1

    run_dir = Path(a.run_root).resolve() / a.name
    if run_dir.exists() and not a.allow_reuse:
        LOG.error("%s 已存在; 要重跑加 --allow-reuse", run_dir)
        return 2

    # ---- 1. 口径 ---------------------------------------------------------- #
    params = param_counts(model_dir)
    budget = plan_budget(params, a.seq, a.batch, a.grad_accum, world_size, a.tokens, a.ratio)
    warmup = a.warmup_steps if a.warmup_steps is not None else max(20, budget["steps"] // 50)

    LOG.info("=" * 78)
    LOG.info("[北京时间 %s] run=%s  模型=%s", bj_now(), a.name, model_dir.name)
    LOG.info("=" * 78)
    LOG.info("模型: d=%d L=%d heads=%s vocab=%d", params["d_model"], params["n_layers"], params["n_heads"], params["vocab"])
    LOG.info("  N_非emb = %s   N_总 = %s", f"{params['n_nonemb']:,}", f"{params['n_total']:,}")
    LOG.info("预算: %s", budget["basis"])
    LOG.info("  tokens = %.4g  (%.2f tok/param)", budget["tokens_actual"], budget["token_param_ratio"])
    LOG.info("  每步 token = %d × %d × %d × %d(GPU) = %s", a.seq, a.batch, a.grad_accum, world_size, f"{budget['tokens_per_step']:,}")
    LOG.info("  步数 = %d   warmup = %d", budget["steps"], warmup)
    LOG.info("  算力 C = 6ND = %.3e FLOPs", budget["flops"])

    if a.dry_run:
        LOG.info("--dry-run: 到此为止")
        print(json.dumps({**params, **budget}, indent=2, ensure_ascii=False))
        return 0

    # ---- 2. 数据 ---------------------------------------------------------- #
    data_dir = run_dir / "data"
    LOG.info("-" * 78)
    LOG.info("备数据: 需要 %.3fB tokens, 从 %s 取", budget["tokens_actual"] / 1e9, a.corpus)
    meta = corpus.prepare(
        corpus_dir=a.corpus,
        data_dir=data_dir,
        tokens_needed=int(budget["tokens_actual"] * 1.02),  # 留 2% 余量
        dataset_name=f"fineweb_{a.name}".replace("-", "_"),
        seed_name=a.name,  # 不同 run 取不同切片
    )
    LOG.info("  切片: %d 个分片(起点 #%d), 可用 %.3fB tokens",
             meta["shards"], meta["shard_offset"], meta["available_tokens"] / 1e9)

    # ---- 3. 配置 ---------------------------------------------------------- #
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    swan = swanlab_config(a, a.name)
    cfg = build_config(
        model_dir=model_dir,
        data_dir=data_dir,
        out_dir=out_dir,
        dataset_name=meta["dataset_name"],
        budget=budget,
        lr=a.lr,
        warmup_steps=warmup,
        grad_checkpointing=a.grad_checkpointing,
        save_final=a.save_final,
        swanlab=swan,
        num_workers=a.num_workers,
        save_steps=a.save_steps,
    )
    yaml_path = run_dir / "train.yaml"
    dump_yaml(cfg, yaml_path)
    (run_dir / "budget.json").write_text(json.dumps({**params, **budget}, indent=2, ensure_ascii=False) + "\n")
    LOG.info("配置已写: %s", yaml_path)

    # ---- 4. 训练 ---------------------------------------------------------- #
    LOG.info("-" * 78)
    LOG.info("[北京时间 %s] 开始训练 (约 %d 步)", bj_now(), budget["steps"])
    t_train = time.time()
    code = run_training(yaml_path, run_dir / "train.log")
    dt = time.time() - t_train
    LOG.info("[北京时间 %s] 训练结束 exit=%d, 用时 %.1fs (%.1f min)", bj_now(), code, dt, dt / 60)
    if code != 0:
        LOG.error("训练失败, 见 %s", run_dir / "train.log")
        return code

    cross_check_params(run_dir, params)

    # ---- 5. 出图 ---------------------------------------------------------- #
    LOG.info("-" * 78)
    try:
        # 传 run_dir 而不是 out_dir: trainer_log.jsonl 在 out/ 下, 但
        # budget.json(每步 token 数)在 run_dir 下, read_curve 两边都会找
        curve = plot_run.read_curve(run_dir)
        plot_run.plot([curve], run_dir / "loss.png", title=f"{a.name}  (N={params['n_total']/1e6:.1f}M, D={budget['tokens_actual']/1e9:.2f}B)")
        s = plot_run.summarize(curve, run_dir)
        LOG.info("loss: 首 %.4f -> 末 %.4f (尾部均值 %.4f, 最小 %.4f)",
                 s["first_loss"], s["last_loss"], s["tail_mean_loss"], s["min_loss"])
        LOG.info("曲线: %s", run_dir / "loss.png")
    except Exception as e:  # 图挂了不该让整个 run 算失败 —— 日志里还有原始数据
        LOG.error("出图失败(%s: %s), 但训练记录在 %s", type(e).__name__, e, out_dir / "trainer_log.jsonl")

    LOG.info("=" * 78)
    LOG.info("完成: %s  总用时 %.1f min", run_dir, (time.time() - t0) / 60)
    LOG.info("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
