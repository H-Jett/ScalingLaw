#!/usr/bin/env python3
"""从 GPT-NeoX 模板派生尺寸阶梯，只生成 config 和 tokenizer。"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass

LOG = logging.getLogger("model_configs")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TEMPLATE = os.path.join(REPO, "models", "templates", "pythia-14m")
DEFAULT_OUT = os.path.join(REPO, "models", "configs")
DEFAULT_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def params_nonemb(d: int, layers: int) -> int:
    """非 embedding 参数量。12Ld² 是权重主体, 13Ld + 2d 是 bias/LayerNorm 的小项。"""
    return 12 * layers * d * d + 13 * layers * d + 2 * d


def params_embed(vocab: int, d: int) -> int:
    """输入 + 输出 embedding(不共享时是 2 份)。"""
    return 2 * vocab * d


def params_total(d: int, layers: int, vocab: int) -> int:
    return params_nonemb(d, layers) + params_embed(vocab, d)


def fmt_n(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))


def ladder_name(total_params: float) -> str:
    """按 pythia 的命名格式给一档起名: 14m / 215m / 1.2b。"""
    m = total_params / 1e6
    if m >= 1000:
        return f"pythia-{round(m / 1000, 1):g}b"
    return f"pythia-{round(m)}m"


@dataclass
class Design:
    """尺寸设计的可调旋钮, 打包传递免得函数签名一长串。"""

    size_by: str = "non-embedding"
    d_granularity: int = 64
    depth_exponent: float = 0.5
    aspect_penalty: float = 0.5
    max_layers: int = 96
    head_dim_min: int = 32
    head_dim_max: int = 128


def ideal_layers(d: int, anchor: tuple[int, int], p: float) -> float:
    """长宽比律给出的"理想层数": L = L0·(d/d0)^p, 锚点取自模板。"""
    d0, l0 = anchor
    return l0 * (d / d0) ** p


def pick_heads(d: int, hd_min: int, hd_max: int) -> tuple[int, int]:
    """选择不小于 d/16 的 2 的幂作为 head_dim，不能整除时向下调整。"""
    want = hd_min
    while want < d / 16 and want < hd_max:
        want *= 2
    hd = want
    while hd >= hd_min:
        if d % hd == 0:
            return d // hd, hd
        hd //= 2
    raise SystemExit(f"d={d} 在 [{hd_min}, {hd_max}] 内没有能整除它的 2 的幂 head_dim")


def search_size(
    target: float,
    *,
    vocab: int,
    anchor: tuple[int, int],
    design: Design,
    d_floor: int = 0,
    n_floor: float = 0.0,
) -> dict:
    """搜索目标参数量和理想长宽比之间的最小加权偏差。"""
    best = None
    for d in range(max(design.d_granularity, d_floor), 8192 + 1, design.d_granularity):
        l_ideal = ideal_layers(d, anchor, design.depth_exponent)
        lo = max(2, int(l_ideal) - 8)
        hi = min(design.max_layers, int(l_ideal) + 8)
        for layers in range(lo, hi + 1):
            n = (
                params_nonemb(d, layers)
                if design.size_by == "non-embedding"
                else params_total(d, layers, vocab)
            )
            if n <= n_floor:
                continue
            score = abs(math.log(n / target)) + design.aspect_penalty * abs(
                math.log(layers / l_ideal)
            )
            if best is None or score < best["score"]:
                n_heads, hd = pick_heads(d, design.head_dim_min, design.head_dim_max)
                best = {
                    "score": score,
                    "d": d,
                    "l": layers,
                    "n": n,
                    "n_heads": n_heads,
                    "head_dim": hd,
                    "l_ideal": l_ideal,
                }
    assert best is not None, f"在约束内找不到接近 {target:.3e} 的解"
    best["target"] = target
    best["nonemb"] = params_nonemb(best["d"], best["l"])
    best["total"] = params_total(best["d"], best["l"], vocab)
    best["emb"] = best["total"] - best["nonemb"]
    return best


def template_spec(base_cfg: dict, vocab: int, *, anchor: tuple[int, int], design: Design) -> dict:
    """第一个尺寸保留模板原样。"""
    d, layers = anchor
    nonemb = params_nonemb(d, layers)
    total = nonemb + params_embed(vocab, d)
    n_heads = base_cfg["num_attention_heads"]
    return {
        "score": 0.0,
        "d": d,
        "l": layers,
        "n_heads": n_heads,
        "head_dim": d // n_heads,
        "l_ideal": ideal_layers(d, anchor, design.depth_exponent),
        "target": float(total if design.size_by == "total" else nonemb),
        "n": float(total if design.size_by == "total" else nonemb),
        "nonemb": nonemb,
        "total": total,
        "emb": total - nonemb,
        "pinned": True,
    }


def build_config(base_cfg: dict, spec: dict, *, seq_len: int | None, dtype: str | None) -> dict:
    """只覆盖 4 个尺寸字段(外加可选的 seq/dtype), 其余全部沿用模板。"""
    cfg = dict(base_cfg)
    cfg["hidden_size"] = spec["d"]
    cfg["num_hidden_layers"] = spec["l"]
    cfg["num_attention_heads"] = spec["n_heads"]
    cfg["intermediate_size"] = 4 * spec["d"]
    if seq_len is not None:
        cfg["max_position_embeddings"] = seq_len
    if dtype is not None:
        cfg["torch_dtype"] = dtype
    return cfg


def link_tokenizer(out_dir: str, template_dir: str, files: tuple[str, ...], copy: bool) -> None:
    """复制或链接 tokenizer；相同的链接可重复生成。"""
    for name in files:
        src, dst = os.path.join(template_dir, name), os.path.join(out_dir, name)
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        if os.path.lexists(dst):
            if os.path.realpath(dst) == os.path.realpath(src):
                continue
            if not copy or os.path.islink(dst):
                raise FileExistsError(f"{dst} 已存在，请使用新的输出目录")
        if copy:
            shutil.copyfile(src, dst)
        else:
            os.symlink(os.path.relpath(src, out_dir), dst)


def measure_total(cfg_dir: str) -> int:
    """在 meta device 上校验 GPT-NeoX 参数量，不加载权重。"""
    import torch
    from transformers import AutoConfig, GPTNeoXForCausalLM

    config = AutoConfig.from_pretrained(cfg_dir)
    if config.model_type != "gpt_neox":
        raise ValueError("仅支持 GPT-NeoX 模板")
    with torch.device("meta"):
        model = GPTNeoXForCausalLM(config)
    return sum(p.numel() for p in model.parameters())


def setup_logging(prefix: str) -> str:
    """日志同时进 stdout 和 logs/<prefix>_<时间戳>.log。"""
    log_dir = os.path.join(REPO, "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        h.setFormatter(fmt)
        LOG.addHandler(h)
    return path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="从模板派生模型尺寸阶梯(只生成 config + tokenizer, 不含权重)"
    )

    ap.add_argument(
        "--template-dir", default=DEFAULT_TEMPLATE, help="模板目录: 从它读 config 与 tokenizer"
    )
    ap.add_argument("--config-name", default="config.json", help="模板 config 文件名")
    ap.add_argument(
        "--tokenizer-files",
        default=",".join(DEFAULT_TOKENIZER_FILES),
        help="要一起产出的 tokenizer 文件, 逗号分隔",
    )
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="输出根目录")

    ap.add_argument("--count", type=int, default=7, help="档数(含最小档)")
    ap.add_argument("--max", type=float, default=1e9, help="最大尺寸(参数量)")
    ap.add_argument(
        "--min", type=float, default=10e6, help="最小尺寸(参数量), 仅 --no-pin-template 时生效"
    )
    ap.add_argument(
        "--pin-template",
        dest="pin_template",
        action="store_true",
        default=True,
        help="第 1 档 = 模板本身原样保留(默认开)",
    )
    ap.add_argument(
        "--no-pin-template",
        dest="pin_template",
        action="store_false",
        help="第 1 档也参与搜索, 阶梯从 --min 起",
    )
    ap.add_argument(
        "--ratio",
        type=float,
        default=0.0,
        help="相邻档位比值; 给了就忽略 --min/--max/--count 的比值推导",
    )
    ap.add_argument(
        "--size-by",
        choices=["non-embedding", "total"],
        default="non-embedding",
        help="阶梯参数量口径",
    )

    ap.add_argument(
        "--d-granularity",
        type=int,
        default=64,
        help="模型宽度的量化步长",
    )
    ap.add_argument(
        "--depth-exponent", type=float, default=0.5, help="长宽比律 L ∝ d^p 的 p(0.5 = 默认)"
    )
    ap.add_argument(
        "--aspect-penalty",
        type=float,
        default=0.5,
        help="长宽比偏差的权重",
    )
    ap.add_argument("--max-layers", type=int, default=96)
    ap.add_argument("--head-dim-min", type=int, default=32, help="head_dim 下限(2 的幂)")
    ap.add_argument("--head-dim-max", type=int, default=128, help="head_dim 上限(2 的幂)")

    ap.add_argument("--seq-len", type=int, default=None, help="覆盖 max_position_embeddings")
    ap.add_argument("--dtype", default=None, help="覆盖 torch_dtype(默认沿用模板)")
    ap.add_argument(
        "--copy-tokenizer", action="store_true", help="真拷贝 tokenizer(默认软链接, 省 ~15MB)"
    )
    ap.add_argument("--verify", dest="verify", action="store_true", default=True)
    ap.add_argument(
        "--no-verify", dest="verify", action="store_false", help="跳过 transformers 实建校验"
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印体积表, 不落盘")
    return ap.parse_args()


def build_ladder(a, base_cfg: dict, vocab: int, design: Design) -> tuple[list[dict], float]:
    """按 --count/--max/--ratio 生成目标序列, 逐档搜出实际尺寸。"""
    anchor = (base_cfg["hidden_size"], base_cfg["num_hidden_layers"])
    if a.pin_template:
        lo = params_total(*anchor, vocab) if design.size_by == "total" else params_nonemb(*anchor)
    else:
        lo = a.min
    if a.ratio > 0:
        r = a.ratio
    elif a.count > 1:
        r = (a.max / lo) ** (1.0 / (a.count - 1))
    else:
        r = 1.0
    targets = [lo * r**i for i in range(a.count)]
    LOG.info("阶梯口径=%s  第1档锚点=%.3e  公比=%.3f", design.size_by, lo, r)
    LOG.info("目标: %s", " → ".join(fmt_n(t) for t in targets))

    specs: list[dict] = []
    for i, t in enumerate(targets):
        if i == 0 and a.pin_template:
            s = template_spec(base_cfg, vocab, anchor=anchor, design=design)
            LOG.info(
                "第 1 档 = 模板本身, 原样保留(d=%d L=%d heads=%d)", s["d"], s["l"], s["n_heads"]
            )
        else:
            prev = specs[-1] if specs else {"d": 0, "n": 0}
            s = search_size(
                t,
                vocab=vocab,
                anchor=anchor,
                design=design,
                d_floor=prev["d"] + design.d_granularity,
                n_floor=prev["n"],
            )
        s["name"] = ladder_name(s["total"])
        specs.append(s)
    return specs, r


def log_table(specs: list[dict], vocab: int) -> None:
    LOG.info("=" * 112)
    LOG.info("尺寸表  (N_非emb = 12Ld² + 13Ld + 2d;  embedding = 2·%d·d)", vocab)
    LOG.info(
        "%-12s %6s %4s %5s %8s %6s | %9s %9s %9s %8s | %7s %8s",
        "目录名",
        "d",
        "L",
        "heads",
        "head_dim",
        "ffw",
        "N_非emb",
        "emb",
        "N_总",
        "emb占比",
        "L/L_ideal",
        "目标偏差",
    )
    for s in specs:
        share = s["emb"] / s["total"]
        dev = "模板" if s.get("pinned") else f"{(s['n'] / s['target'] - 1) * 100:+.1f}%"
        LOG.info(
            "%-12s %6d %4d %5d %8d %6d | %9s %9s %9s %7.1f%%%s | %7.3f %8s",
            s["name"],
            s["d"],
            s["l"],
            s["n_heads"],
            s["head_dim"],
            4 * s["d"],
            fmt_n(s["nonemb"]),
            fmt_n(s["emb"]),
            fmt_n(s["total"]),
            share * 100,
            " ⚠" if share > 0.5 else "",
            s["l"] / s["l_ideal"],
            dev,
        )
    LOG.info("=" * 112)
    LOG.info("注: 目录名用**总参数量**口径(与 pythia 官方命名一致); N_非emb 才是计算层参数")
    if any(s["emb"] / s["total"] > 0.5 for s in specs):
        LOG.warning("有档位 embedding 占比 >50% —— 主要参数是词表, 不是计算层")


def main() -> None:
    a = parse_args()
    log_path = setup_logging("generate_model_configs")
    LOG.info("日志: %s", log_path)
    LOG.info("启动参数: %s", vars(a))

    template_dir = os.path.abspath(a.template_dir)
    out_root = os.path.abspath(a.out_dir)
    cfg_path = os.path.join(template_dir, a.config_name)
    if not os.path.exists(cfg_path):
        raise SystemExit(
            f"找不到模板 config: {cfg_path}\n"
            f"模板目录应含 {a.config_name} 与 [{a.tokenizer_files}]。\n"
            "模板下载是一次性操作, 需要时手动跑一次:\n"
            "  HF_TOKEN=... hf download EleutherAI/pythia-14m "
            "--local-dir models/templates/pythia-14m"
        )

    with open(cfg_path) as f:
        base_cfg = json.load(f)
    vocab = base_cfg["vocab_size"]
    anchor = (base_cfg["hidden_size"], base_cfg["num_hidden_layers"])
    design = Design(
        size_by=a.size_by,
        d_granularity=a.d_granularity,
        depth_exponent=a.depth_exponent,
        aspect_penalty=a.aspect_penalty,
        max_layers=a.max_layers,
        head_dim_min=a.head_dim_min,
        head_dim_max=a.head_dim_max,
    )
    LOG.info(
        "模板: %s  vocab=%d  d=%d  L=%d  heads=%d  ffw=%d  seq=%d  tie=%s",
        cfg_path,
        vocab,
        *anchor,
        base_cfg["num_attention_heads"],
        base_cfg["intermediate_size"],
        base_cfg["max_position_embeddings"],
        base_cfg.get("tie_word_embeddings"),
    )
    LOG.info("长宽比律: L = %d·(d/%d)^%g  (锚点 = 模板)", anchor[1], anchor[0], a.depth_exponent)

    specs, _ = build_ladder(a, base_cfg, vocab, design)
    log_table(specs, vocab)

    if a.dry_run:
        LOG.info("[dry-run] 未落盘。")
        return

    tokenizer_files = tuple(x.strip() for x in a.tokenizer_files.split(",") if x.strip())
    for s in specs:
        out_dir = os.path.join(out_root, s["name"])
        os.makedirs(out_dir, exist_ok=True)
        cfg = build_config(base_cfg, s, seq_len=a.seq_len, dtype=a.dtype)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        link_tokenizer(out_dir, template_dir, tokenizer_files, a.copy_tokenizer)
        LOG.info(
            "已写出: %s  (d=%d L=%d heads=%d ffw=%d)",
            out_dir,
            s["d"],
            s["l"],
            s["n_heads"],
            4 * s["d"],
        )

    if a.verify:
        LOG.info("-" * 100)
        LOG.info("实建校验(在 meta device 上建模型数参数, 不占显存)")
        LOG.info("%-12s %14s %14s %10s", "目录名", "公式总参", "实建总参", "误差")
        bad = 0
        for s in specs:
            try:
                real = measure_total(os.path.join(out_root, s["name"]))
            except Exception as e:  # noqa: BLE001
                LOG.warning("%-12s 实建失败: %s", s["name"], e)
                bad += 1
                continue
            diff = real - s["total"]
            bad += diff != 0
            LOG.info("%-12s %14s %14s %10d", s["name"], f"{s['total']:,}", f"{real:,}", diff)
        LOG.info("校验: %s", "全部一致 ✅" if bad == 0 else f"{bad} 个不一致 ❌")

    LOG.info("完成: %d 个模型目录 -> %s", len(specs), out_root)
    LOG.info("tokenizer: %s", "真拷贝" if a.copy_tokenizer else "软链接(指向模板)")


if __name__ == "__main__":
    main()
