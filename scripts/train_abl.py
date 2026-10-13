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
import shutil
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
    vocab_emb = vocab * d  # 单个 embedding 矩阵(输入或 LM head)
    total = nonemb + 2 * vocab_emb
    return {
        "d_model": d,
        "n_layers": layers,
        "n_heads": cfg.get("num_attention_heads"),
        "vocab": vocab,
        "n_nonemb": nonemb,
        "n_embed": 2 * vocab_emb,
        "n_vocab_emb": vocab_emb,  # 只用于解释 HF 那个中间量, 不参与口径计算
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
        # Chinchilla 的 N 是**含 embedding** 的总参数量 —— Appendix F 原文:
        #   "We include all training FLOPs, including those contributed to by the
        #    embedding matrices... we also count embeddings matrices in the total
        #    parameter count."
        # 所以 "20 tok/param" 的 20 乘在 N_total 上。(注意: 按**非** embedding 参数量
        # 算是 Kaplan et al. 2020 的口径, 两者常被搞混。)
        # ⚠️ 但我们的 vocab=50304 偏大, 小端 embedding 占比极高(pythia-14m 占 91.5%),
        # 此时 ratio×N_total 会给出与"计算层"严重脱节的 D —— 见下面的告警。
        tokens = ratio * params["n_total"]
        basis = f"ratio={ratio} × N_total"
        emb_share = params["n_embed"] / params["n_total"]
        if emb_share > 0.5:
            LOG.warning(
                "embedding 占 N_total 的 %.1f%%: ratio=%g × N_total 给出的 D 主要是在"
                "喂词表, 计算层的实际 tok/param 只有 %.1f。论文的 20 tok/param 建立在"
                "embedding 占比很低的前提上, 在 d=%d 这一档不成立 —— 定网格时留意。",
                100 * emb_share, ratio, tokens / params["n_nonemb"], params["d_model"])
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
        # 两个算力口径都记下来, 供 §3 的口径敏感性分析用(见 PLAN §3 的 ⚠️):
        #   flops        = 6·N_total·D   ← 论文口径(把 embedding 也算进 N)
        #   flops_nonemb = 6·N_nonemb·D  ← 彻底排除 embedding 的口径
        # 两者在小模型上差别巨大(14m: embedding 占 91.5%), 而大模型上几乎一致
        # (1.2b: 15.9%) —— 所以"用哪个口径"会系统性地改变谷底位置, 必须两个都算。
        # ⚠️ 千万别用 HF 的 total_flos, 它不是这两个中的任何一个, 见 log_flops_calibers。
        "flops": 6 * params["n_total"] * steps * tps,
        "flops_nonemb": 6 * params["n_nonemb"] * steps * tps,
        # 两个 tok/param 都记 —— 小端两者差一个数量级(14m: 5.7 vs 67),
        # 报告结论时必须说清用的是哪个, 否则不可比。
        "token_param_ratio": (steps * tps) / params["n_total"],
        "token_param_ratio_nonemb": (steps * tps) / params["n_nonemb"],
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


def log_flops_calibers(run_dir: Path, params: dict, budget: dict) -> None:
    """把我们自己算的两个算力口径打出来, 并明确指出 HF 的 total_flos 不在其中。

    ## 两个口径都由我们自己算, 不依赖任何库

        C_total = 6·N_total·D    ← 论文口径。论文明确 "include all training FLOPs,
                                   including those contributed to by the embedding
                                   matrices", 所以 embedding 算进 N
        C_nonemb = 6·N_nonemb·D  ← 彻底排除 embedding 的口径(输入 embedding 和
                                   LM head 都不算)

    ## ⚠️ 为什么不能用 `num_parameters(exclude_embeddings=True)`

    HF 的 `total_flos` 用的是 `6 × num_parameters(exclude_embeddings=True) × tokens`
    (transformers 5.8.0, `trainer.py:3900`)。但实测在 `tie_word_embeddings=False`
    时, `exclude_embeddings=True` **只排除了输入 embedding, LM head 仍被算进去**:

        num_parameters()                          = 14,067,712   ← N_total, 与公式一致
        num_parameters(exclude_embeddings=True)   =  7,628,800   ← 排了 1 个 embedding
        真正的 N_nonemb                            =  1,189,888   ← 排了 2 个

    所以它是个**中间量**(= N_total − vocab·d), 既不是 C_total 也不是 C_nonemb。
    危害在于偏差随尺寸变化 —— 实测比值 HF/C_total 从 14m 的 0.542 升到 1.2b 的
    0.920(因为 vocab=50304 偏大, embedding 占比从 91.5% 降到 15.9%), 差 1.7 倍。
    拿它当横轴会把小模型系统性压向一侧, 直接扭曲指数 a, 且因为是尺寸相关的偏差、
    不是常数偏移, 事后很难发现。

    本函数因此**自己拆参数量算**, 完全绕开 HF 那个数; 只在日志里对照打印,
    便于发现上游行为变化。
    """
    c_total = budget["flops"]
    c_nonemb = budget["flops_nonemb"]
    LOG.info("算力口径(均自算, 不依赖库):")
    LOG.info("  C_total  = 6·N_total·D  = 6 × %s × %s = %.4e", f"{params['n_total']:,}", f"{budget['tokens_actual']:,}", c_total)
    LOG.info("  C_nonemb = 6·N_nonemb·D = 6 × %s × %s = %.4e", f"{params['n_nonemb']:,}", f"{budget['tokens_actual']:,}", c_nonemb)
    LOG.info("  两者之比 = %.4f  (embedding 占 N_total 的 %.1f%%)", c_nonemb / c_total, 100 * params["n_embed"] / params["n_total"])
    LOG.info(
        "  → 拟合时建议两个口径各跑一遍(见 PLAN §3 的敏感性分析): 小模型端 %.1f%% 的参数是词表,"
        " 论文『embedding 贡献很小』的前提在 d=%d 时不成立",
        100 * params["n_embed"] / params["n_total"], params["d_model"],
    )

    res = None
    for cand in [run_dir / "train_results.json", *sorted(run_dir.glob("*/train_results.json"))]:
        if cand.exists():
            res = json.loads(cand.read_text())
            break
    if res and res.get("total_flos"):
        hf = float(res["total_flos"])
        LOG.info(
            "  对照: HF total_flos = %.4e  →  HF/C_total = %.3f (即 N_total − vocab·d = %s, 中间量, 不用)",
            hf, hf / c_total, f"{params['n_total'] - params['n_vocab_emb']:,}",
        )


# 训练完成后删掉的"体积大但拟合缩放律用不到"的东西。
# 保留: trainer_log.jsonl(loss 的唯一必需来源)、trainer_state.json(含 grad_norm)、
#       loss.png/summary.json(我们生成的)、train.yaml/budget.json(口径)。
# 删除: 权重、分词器副本、被重写过的 config —— 拟合只用 loss-vs-C, 复跑用 train.yaml。
PRUNE_PATTERNS = (
    "model.safetensors",
    "*.safetensors",
    "*.bin",  # pytorch_model.bin / training_args.bin
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "config.json",
    "README.md",
    "all_results.json",  # train_results.json 的孪生副本(md5 相同)
)


def prune_run_artifacts(run_dir: Path, keep: bool) -> None:
    """清掉本 run 里与缩放律无关的大件(权重等), 只留 loss / 口径 / 配置。

    为什么必须删: `pt/workflow.py:64` 的 `trainer.save_model()` 是无条件执行的,
    没有开关能跳过(LlaMA-Factory 未 override, HF 原生 save_model 总是写盘)。
    而 `/volume/data` 配额只剩几百 M —— 14m 的权重就 54M, 1.2b 要 4.8G,
    不删的话网格跑到一半必然撑爆。

    `--keep-model` 可以关掉这个清理(比如想拿模型去做别的验证或推理)。
    """
    out_dir = run_dir / "out"
    if not out_dir.exists():
        return
    if keep:
        LOG.info("--keep-model: 保留权重等全部产物")
        return

    freed = 0
    removed: list[str] = []
    for pattern in PRUNE_PATTERNS:
        for f in out_dir.rglob(pattern):
            if f.is_file() and not f.is_symlink():
                size = f.stat().st_size
                freed += size
                removed.append(f"{f.relative_to(run_dir)} ({size / 1e6:.1f}M)")
                f.unlink()

    # 中间检查点目录(HF 的 checkpoint-N/, 含逐卡 rng_state)
    for ckpt in sorted(out_dir.glob("checkpoint-*")):
        if ckpt.is_dir():
            size = sum(p.stat().st_size for p in ckpt.rglob("*") if p.is_file())
            freed += size
            removed.append(f"{ckpt.relative_to(run_dir)}/ 目录 ({size / 1e6:.1f}M)")
            shutil.rmtree(ckpt)

    LOG.info("清理: 删掉 %d 项, 释放 %.1fM(只保留 loss / 配置 / 口径)", len(removed), freed / 1e6)
    for r in removed:
        LOG.info("  - %s", r)


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
    g.add_argument("--ratio", type=float,
                   help="训练 token 数 = ratio × N_total (Chinchilla 的 20 tok/param 就是这个口径, "
                        "论文把 embedding 算进 N)。⚠️ 小端 embedding 占比高时会告警")
    ap.add_argument("--seq", type=int, default=2048, help="序列长度; 实测 2048 时 MFU 最高(见 PLAN §1.1)")
    ap.add_argument("--batch", type=int, default=8, help="每卡 micro batch")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--world-size", type=int, default=None, help="GPU 数, 默认自动探测")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-steps", type=int, default=None, help="默认 max(20, 2%% of steps)")
    ap.add_argument("--grad-checkpointing", action="store_true", help="大模型放不下时开")
    ap.add_argument("--save-final", action="store_true", help="存最后一个检查点(默认不存, 拟合只需 loss)")
    ap.add_argument("--keep-model", action="store_true", help="训练后保留权重(默认删掉, 只留 loss/口径/配置)")
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
    log_flops_calibers(run_dir, params, budget)

    # ---- 5. 出图(必须在清理之前: 图/摘要依赖 trainer_log.jsonl, 清理不删它,
    #          但顺序上先出图更稳 —— 万一以后清理规则改宽了也不会误伤)----- #
    LOG.info("-" * 78)
    try:
        # 传 run_dir 而不是 out_dir: trainer_log.jsonl 在 out/ 下, 但
        # budget.json(每步 token 数)在 run_dir 下, read_curve 两边都会找
        curve = plot_run.read_curve(run_dir)
        # 标题带上两种 N —— 小端两者差一个数量级, 光看一个数容易误读这个 run 的大小
        plot_run.plot([curve], run_dir / "loss.png",
                      title=f"{a.name}  (N_total={params['n_total']/1e6:.1f}M, "
                            f"N_nonemb={params['n_nonemb']/1e6:.1f}M, "
                            f"D={budget['tokens_actual']/1e9:.2f}B)")
        s = plot_run.summarize(curve, run_dir)
        LOG.info("loss: 首 %.4f -> 末 %.4f (尾部均值 %.4f, 最小 %.4f)",
                 s["first_loss"], s["last_loss"], s["tail_mean_loss"], s["min_loss"])
        LOG.info("曲线: %s", run_dir / "loss.png")
    except Exception as e:  # 图挂了不该让整个 run 算失败 —— 日志里还有原始数据
        LOG.error("出图失败(%s: %s), 但训练记录在 %s", type(e).__name__, e, out_dir / "trainer_log.jsonl")

    # ---- 6. 清理 ---------------------------------------------------------- #
    LOG.info("-" * 78)
    prune_run_artifacts(run_dir, keep=a.keep_model)
    LOG.info("run 目录现在的大小: %s", subprocess.run(["du", "-sh", str(run_dir)], capture_output=True, text=True).stdout.split()[0])

    LOG.info("=" * 78)
    LOG.info("完成: %s  总用时 %.1f min", run_dir, (time.time() - t0) / 60)
    LOG.info("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
