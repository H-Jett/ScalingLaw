#!/usr/bin/env python3
"""独立训练入口：读取已准备的数据、计算预算、生成配置并训练。"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import yaml

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "logs"
RUN_ROOT = REPO / "runs"

LOG = logging.getLogger("train")


@dataclass(frozen=True)
class DistributedEnv:
    node_rank: int
    nnodes: int
    nproc_per_node: int
    master_addr: str
    master_port: int

    @property
    def world_size(self) -> int:
        """实际训练进程总数（用于全局 token 预算）。"""
        return self.nnodes * self.nproc_per_node

    def as_dict(self) -> dict:
        return {**asdict(self), "world_size": self.world_size}

    def launch_env(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if environ is None else environ)

        for key in ("RANK", "WORLD_SIZE"):
            env.pop(key, None)
        env.update(
            DISABLE_VERSION_CHECK="1",
            FORCE_TORCHRUN="1",
            NNODES=str(self.nnodes),
            NODE_RANK=str(self.node_rank),
            NPROC_PER_NODE=str(self.nproc_per_node),
            MASTER_ADDR=self.master_addr,
            MASTER_PORT=str(self.master_port),
        )
        return env


def resolve_distributed(
    world_size_override=None, world_size_unit="nodes", environ=None
) -> DistributedEnv:
    """节点口径：RANK=节点编号；进程口径：RANK=本节点首个进程编号。"""
    env = os.environ if environ is None else environ
    if "LOCAL_RANK" in env:
        raise ValueError("请每节点直接执行 train.sh 一次，不要用 torchrun 启动它")
    if world_size_unit not in ("nodes", "processes"):
        raise ValueError("WORLD_SIZE 口径必须是 nodes 或 processes")
    if world_size_override is not None and world_size_override <= 0:
        raise ValueError("--world-size 必须为正")
    if "NPROC_PER_NODE" in env:
        nproc = int(env["NPROC_PER_NODE"])
    elif world_size_override is not None and not any(
        k in env for k in ("RANK", "WORLD_SIZE", "NNODES", "NODE_RANK")
    ):
        nproc = world_size_override
    else:
        import torch

        nproc = torch.cuda.device_count() or 1
    if nproc <= 0:
        raise ValueError("NPROC_PER_NODE 必须为正")

    factor = nproc if world_size_unit == "processes" else 1
    size = int(env.get("WORLD_SIZE", int(env.get("NNODES", 1)) * factor))
    rank = int(env.get("RANK", 0))
    if size <= 0 or rank < 0 or size % factor or rank % factor:
        raise ValueError("WORLD_SIZE 或 RANK 与每节点进程数不匹配")
    nnodes, node_rank = size // factor, rank // factor
    if "NNODES" in env and int(env["NNODES"]) != nnodes:
        raise ValueError("NNODES 与 WORLD_SIZE 不一致")
    if "NODE_RANK" in env:
        explicit_rank = int(env["NODE_RANK"])
        if "RANK" in env and explicit_rank != node_rank:
            raise ValueError("NODE_RANK 与 RANK 不一致")
        node_rank = explicit_rank
    if not 0 <= node_rank < nnodes:
        raise ValueError("节点编号超出范围")
    if world_size_override is not None and world_size_override != nnodes * nproc:
        raise ValueError("--world-size 与环境推导的全局进程数不一致")
    addr = env.get("MASTER_ADDR", "127.0.0.1").strip()
    port = int(env.get("MASTER_PORT", 29500))
    if not addr or (nnodes > 1 and addr in ("localhost", "127.0.0.1", "::1")):
        raise ValueError("多节点训练必须设置可访问的 MASTER_ADDR")
    if not 1 <= port <= 65535:
        raise ValueError("MASTER_PORT 必须在 1～65535 之间")
    return DistributedEnv(node_rank, nnodes, nproc, addr, port)


def param_counts(model_dir: Path) -> dict:
    """GPT-NeoX 参数量：计算层、输入 embedding 和输出 head。"""
    cfg = json.loads((model_dir / "config.json").read_text())
    d, layers, vocab = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    nonemb = 12 * layers * d * d + 13 * layers * d + 2 * d
    embedding = vocab * d
    return {
        "d_model": d,
        "n_layers": layers,
        "n_heads": cfg["num_attention_heads"],
        "vocab": vocab,
        "n_nonemb": nonemb,
        "n_vocab_emb": embedding,
        "n_embed": 2 * embedding,
        "n_total": nonemb + 2 * embedding,
    }


def tokens_per_step(seq: int, batch: int, grad_accum: int, world_size: int) -> int:
    """每步 token 数包含所有训练进程。"""
    return seq * batch * grad_accum * world_size


def plan_budget(params, seq, batch, grad_accum, world_size, tokens, ratio) -> dict:
    """按全局 batch 取整步数，记录两种参数和算力口径。"""
    if min(seq, batch, grad_accum, world_size) <= 0:
        raise ValueError("序列长度、batch、梯度累积和进程数必须为正")
    if tokens is None:
        if ratio is None:
            raise ValueError("必须指定 tokens 或 ratio")
        tokens = ratio * params["n_total"]
        basis = f"ratio={ratio} × N_total"
        if params["n_embed"] > params["n_total"] / 2:
            LOG.warning(
                "embedding 占总参数的 %.1f%%，ratio 使用总参数口径",
                100 * params["n_embed"] / params["n_total"],
            )
    else:
        basis = f"tokens={tokens:.3e}"
    if not math.isfinite(tokens) or tokens <= 0:
        raise ValueError("tokens 必须是有限正数")
    tps = tokens_per_step(seq, batch, grad_accum, world_size)
    steps = max(1, round(tokens / tps))
    actual = steps * tps
    return {
        "seq_len": seq,
        "micro_batch": batch,
        "grad_accum": grad_accum,
        "world_size": world_size,
        "tokens_per_step": tps,
        "steps": steps,
        "tokens_requested": tokens,
        "tokens_actual": actual,
        "basis": basis,
        "n_total": params["n_total"],
        "n_nonemb": params["n_nonemb"],
        "flops": 6 * params["n_total"] * actual,
        "flops_nonemb": 6 * params["n_nonemb"] * actual,
        "token_param_ratio": actual / params["n_total"],
        "token_param_ratio_nonemb": actual / params["n_nonemb"],
    }


def build_config(args, budget, model_dir, data_dir, out_dir, dataset_name) -> dict:
    """生成训练配置，cosine 周期与预算步数一致。"""
    config = {
        "model_name_or_path": str(model_dir),
        "trust_remote_code": True,
        "train_from_scratch": True,
        "stage": "pt",
        "finetuning_type": "full",
        "do_train": True,
        "dataset": dataset_name,
        "dataset_dir": str(data_dir),
        "template": "default",
        "cutoff_len": budget["seq_len"],
        "packing": True,
        "preprocessing_num_workers": args.num_workers,
        "overwrite_cache": True,
        "output_dir": str(out_dir),
        "overwrite_output_dir": True,
        "report_to": "tensorboard",
        "logging_steps": 1,
        "plot_loss": False,
        "save_strategy": "steps" if args.save_final else "no",
        "per_device_train_batch_size": budget["micro_batch"],
        "gradient_accumulation_steps": budget["grad_accum"],
        "learning_rate": args.lr,
        "max_steps": budget["steps"],
        "lr_scheduler_type": "cosine",
        "warmup_steps": args.warmup_steps,
        "bf16": True,
        "gradient_checkpointing": args.grad_checkpointing,
        "ddp_timeout": 180000000,
    }
    if args.save_final:
        config.update(save_steps=args.save_steps or budget["steps"], save_total_limit=1)
    config.update(swanlab_config(args, args.name))
    return config


def dump_yaml(config: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


def swanlab_config(args: argparse.Namespace, run_name: str) -> dict:
    """配置可选的 swanlab 日志。"""
    mode = args.swanlab_mode
    if mode == "off":
        return {}
    key = os.environ.get("SWANLAB_API_KEY")
    if mode == "cloud" and not key:
        LOG.warning("缺少 SWANLAB_API_KEY，使用本地日志")
        mode = "local"
    config = {
        "use_swanlab": True,
        "swanlab_project": args.swanlab_project,
        "swanlab_run_name": run_name,
        "swanlab_mode": mode,
    }
    if mode == "local":
        config["swanlab_logdir"] = str(Path(args.run_root) / "_swanlab")
    else:
        config["swanlab_api_key"] = key
    return config


_LOG_MARKERS = (
    "train_loss",
    "trainable params",
    "[INFO|",
    "[WARNING|",
    "[ERROR|",
    "Traceback (most recent call last)",
)


def run_training(yaml_path: Path, log_path: Path, distributed: DistributedEnv) -> int:
    """跑训练, stdout 同时进控制台和日志文件。返回退出码。"""
    env = distributed.launch_env()
    proc = subprocess.Popen(
        [sys.executable, "-m", "llamafactory.cli", "train", str(yaml_path)],
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

            if any(k in line for k in _LOG_MARKERS):
                LOG.info("  | %s", line.rstrip())
    return proc.wait()


def wait_for_config(run_dir: Path, launch_spec: dict, timeout: float) -> None:
    """共享文件系统上等待主节点完成配置，再让本节点加入训练。"""
    ready = run_dir / "launch.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready.exists():
            try:
                actual = json.loads(ready.read_text())
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            if actual != launch_spec:
                raise ValueError("本节点的参数/分布式配置与主节点 launch.json 不一致")
            if not (run_dir / "train.yaml").is_file():
                raise ValueError("主节点已发布 launch.json，但 train.yaml 不存在")
            return
        time.sleep(1)
    raise TimeoutError(
        f"等待主节点配置超时（{timeout:g}s）：{ready}；所有节点必须共享 run-root 和语料路径"
    )


def cross_check_params(run_dir: Path, params: dict) -> None:
    """用框架日志核对总参数量。"""
    match = re.search(
        r"trainable params:\s*([\d,]+)", (run_dir / "train.log").read_text(errors="ignore")
    )
    if not match:
        LOG.warning("框架日志缺少参数量，无法复核")
        return
    actual = int(match[1].replace(",", ""))
    if actual != params["n_total"]:
        raise ValueError(f"参数量不一致：实际 {actual}，预算 {params['n_total']}")
    LOG.info("参数量复核通过：%s", actual)


def log_flops_calibers(run_dir: Path, params: dict, budget: dict) -> None:
    """记录两种算力口径；HF total_flos 仅用于对照。"""
    LOG.info("C_total=%.4e C_nonemb=%.4e", budget["flops"], budget["flops_nonemb"])
    path = run_dir / "out" / "train_results.json"
    if path.exists():
        hf_flops = json.loads(path.read_text()).get("total_flos")
        if hf_flops:
            LOG.info("HF/C_total=%.3f（不用于拟合）", hf_flops / budget["flops"])


PRUNE_PATTERNS = (
    "model.safetensors",
    "*.safetensors",
    "*.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "config.json",
    "README.md",
    "all_results.json",
)


def prune_run_artifacts(run_dir: Path, keep: bool) -> None:
    """保留训练记录；keep=True 时也保留权重。"""
    out = run_dir / "out"
    if keep or not out.exists():
        return
    freed = 0
    for pattern in PRUNE_PATTERNS:
        for path in out.rglob(pattern):
            if path.is_file() and not path.is_symlink():
                freed += path.stat().st_size
                path.unlink()
    for path in out.glob("checkpoint-*"):
        if path.is_dir() and not path.is_symlink():
            freed += sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            shutil.rmtree(path)
    LOG.info("整理训练产物，释放 %.1f MB", freed / 1e6)


def setup_logging(run_name: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG.handlers.clear()
    LOG.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(LOG_DIR / f"{run_name}.log")):
        handler.setFormatter(formatter)
        LOG.addHandler(handler)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="独立训练：读取已准备的数据 -> 生成配置 -> 训练")
    ap.add_argument("--model", required=True, help="模型 config 目录, 如 models/configs/pythia-86m")
    ap.add_argument("--name", required=True, help="run 名, 决定 runs/<name>/ 和 swanlab run 名")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tokens", type=float, help="训练 token 数(显式)")
    g.add_argument(
        "--ratio",
        type=float,
        help="训练 token 数 = ratio × 总参数量",
    )
    ap.add_argument("--seq", type=int, default=2048, help="序列长度")
    ap.add_argument("--batch", type=int, default=8, help="每卡 micro batch")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument(
        "--world-size", type=int, default=None, help="全局训练进程数；有平台环境变量时必须与其一致"
    )
    ap.add_argument(
        "--world-size-unit",
        choices=["nodes", "processes"],
        default="nodes",
        help="环境变量 WORLD_SIZE 的口径：nodes=节点数（默认），processes=全局进程数",
    )
    ap.add_argument(
        "--config-wait-timeout",
        type=float,
        default=1800,
        help="非主节点等待共享配置的秒数，默认 1800",
    )
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-steps", type=int, default=None, help="默认按预算选择，最多 steps-1")
    ap.add_argument("--grad-checkpointing", action="store_true", help="大模型放不下时开")
    ap.add_argument(
        "--save-final", action="store_true", help="存最后一个检查点(默认不存, 拟合只需 loss)"
    )
    ap.add_argument(
        "--keep-model", action="store_true", help="训练后保留权重(默认删掉, 只留 loss/口径/配置)"
    )
    ap.add_argument("--save-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--data-dir", default=None, help="corpus.py 已准备的数据目录，默认 runs/<name>/data"
    )
    ap.add_argument(
        "--dataset", default=None, help="dataset_info.json 中的数据集名，默认 fineweb_<name>"
    )
    ap.add_argument("--run-root", default=str(RUN_ROOT))
    ap.add_argument("--swanlab-mode", default="local", choices=["local", "cloud", "off"])
    ap.add_argument("--swanlab-project", default="scalinglaw")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划, 不训练")
    ap.add_argument("--allow-reuse", action="store_true", help="run 目录已存在时覆盖重跑")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        distributed = resolve_distributed(args.world_size, args.world_size_unit)
        if args.config_wait_timeout <= 0:
            raise ValueError("配置等待时间必须为正")
        if distributed.nnodes > 1 and args.allow_reuse:
            raise ValueError("多节点训练请使用新 run 名")
        model_dir = Path(args.model).resolve()
        params = param_counts(model_dir)
        budget = plan_budget(
            params,
            args.seq,
            args.batch,
            args.grad_accum,
            distributed.world_size,
            args.tokens,
            args.ratio,
        )
        if args.warmup_steps is None:
            args.warmup_steps = min(budget["steps"] - 1, max(20, budget["steps"] // 50))
        if not 0 <= args.warmup_steps < budget["steps"]:
            raise ValueError("warmup_steps 必须在 [0, steps) 内")
    except (ValueError, FileNotFoundError) as exc:
        LOG.error("%s", exc)
        return 2

    rank = distributed.node_rank
    setup_logging(args.name if rank == 0 else f"{args.name}.node-{rank}")
    run_dir = Path(args.run_root).resolve() / args.name
    data_dir = Path(args.data_dir).resolve() if args.data_dir else run_dir / "data"
    dataset_name = args.dataset or f"fineweb_{args.name}".replace("-", "_")
    artifacts = ("train.yaml", "launch.json", "budget.json", "out")
    if rank == 0 and not args.allow_reuse and any((run_dir / name).exists() for name in artifacts):
        LOG.error("%s 已有训练产物；重跑需 --allow-reuse", run_dir)
        return 2
    plan = {
        **params,
        **budget,
        "distributed": distributed.as_dict(),
        "run_name": args.name,
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "dataset_name": dataset_name,
    }
    LOG.info(
        "run=%s node=%d/%d steps=%d warmup=%d",
        args.name,
        rank,
        distributed.nnodes,
        budget["steps"],
        args.warmup_steps,
    )
    LOG.info(
        "每步 token = %d × %d × %d × %d(GPU) = %s",
        args.seq,
        args.batch,
        args.grad_accum,
        distributed.world_size,
        budget["tokens_per_step"],
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    launch_spec = {
        "arguments": vars(args),
        "model_dir": str(model_dir),
        "run_dir": str(run_dir),
        "distributed": {k: v for k, v in distributed.as_dict().items() if k != "node_rank"},
    }
    try:
        if rank == 0:
            info = json.loads((data_dir / "dataset_info.json").read_text())
            if dataset_name not in info:
                raise ValueError(f"数据目录没有数据集 {dataset_name}")
            metadata = data_dir / "corpus.json"
            if (
                metadata.exists()
                and json.loads(metadata.read_text())["available_tokens"] < budget["tokens_actual"]
            ):
                raise ValueError("已准备的数据不足")
            out_dir = run_dir / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            config = build_config(args, budget, model_dir, data_dir, out_dir, dataset_name)
            dump_yaml(config, run_dir / "train.yaml")
            (run_dir / "budget.json").write_text(
                json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
            )
            (run_dir / "launch.json").write_text(
                json.dumps(launch_spec, indent=2, ensure_ascii=False) + "\n"
            )
        else:
            wait_for_config(run_dir, launch_spec, args.config_wait_timeout)
        log_name = "train.log" if rank == 0 else f"train.node-{rank}.log"
        code = run_training(run_dir / "train.yaml", run_dir / log_name, distributed)
        if code or rank != 0:
            return code
        cross_check_params(run_dir, params)
        log_flops_calibers(run_dir, params, budget)
        prune_run_artifacts(run_dir, keep=args.keep_model)
    except (ValueError, FileNotFoundError, TimeoutError) as exc:
        LOG.error("%s", exc)
        return 2
    LOG.info("训练完成：%s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
