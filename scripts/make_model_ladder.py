#!/usr/bin/env python3
"""从 HF 的 pythia-14m 模板派生一整条模型尺寸阶梯(把模板本身当最小档, 依次向上到 1B)。

做四件事:
  1. 下载 pythia-14m 的 config/tokenizer 到 models/templates/pythia-14m/
     (官方源优先, 失败自动切 hf-mirror; 已存在则跳过)
  2. **把模板本身原样作为第 1 档**(不参与搜索 —— 它是整条阶梯的锚点, 也是唯一一个
     有官方权重可对照的尺寸), 然后从它往上依次派生出其余档位。每档只改 4 个尺寸字段
     (hidden_size / num_hidden_layers / num_attention_heads / intermediate_size)
     → models/configs/pythia-<size>/
  3. 打印尺寸表: 每个尺寸用公式算参数量, 并**实际在 meta device 上建一次模型**校验
     (不占显存, 也不下载权重)
  4. 把尺寸表 + 尺寸设计规则写成 models/configs/README.md, 便于随时查阅

产出的模型目录只有 config.json + tokenizer, 没有权重 —— 供 LLaMA-Factory 的
`train_from_scratch: true` 从零初始化。

================================================================================
尺寸设计规则(想改模型怎么放大/缩小, 读这一段就够了)
================================================================================
[1] 骨架: 参数量几乎只由 d 和 L 决定
        N_非emb ≈ 12·L·d²        每层 = 4d²(注意力 QKV+O) + 8d²(MLP, ffw=4d)
    精确式(已对 pythia 全家族校验, 见 --self-test):
        N_非emb = 12·L·d² + 13·L·d + 2·d
        N_总    = N_非emb + 2·vocab·d      (embedding 输入/输出不共享, tie=False)
    注意 d 是平方项: d 翻倍 → 参数 ×4; L 翻倍 → 参数 ×2。所以"放大模型"主要是放大 d。

[2] 长宽比律: d 和 L 必须一起长, 且 d 长得更快
        L ∝ d^0.5   等价于   d ∝ N^0.4,  L ∝ N^0.2
    这是从 pythia 家族两个端点实测标定出来的(14m: d=128/L=6 → 12b: d=5120/L=36,
    用非 embedding 参数量算, 指数 0.403 / 0.196)。只加宽不加高 → 学不会组合;
    只加高不加宽 → 梯度信号衰减、难训。用 --depth-exponent 可调(0.5 = 默认)。

[3] 硬约束(离散化, 这是"想得到但拿不到"的根源):
        d  必须是 64 的倍数       → tensor core 对齐 (--d-granularity)
        head_dim = d / n_heads 取 clamp(≥d/16 的最小 2 的幂, 32, 128)
                                  → 「头数随 d 缓慢增长」(见 pick_heads)
        ffw = 4·d                 → pythia/Chinchilla 一致惯例
        L   是整数
    因为 d 被量化(且必须逐档递增), **不可能同时精确命中目标参数量和理想长宽比**;
    脚本用 --aspect-penalty 在两个偏差之间做加权折中, 并把两个偏差都打进表里。

[4] 尺寸口径(最容易踩的坑): "10M 模型"到底指什么?
    vocab=50304 时, embedding 占 2·50304·d = 100608·d 个参数。d=256 时这张查找表
    就是 25.8M —— 比整个"10M 模型"还大。所以:
      --size-by total          : 总参数量(HF/pythia 命名口径)。小端会被 embedding 淹没,
                                 且 V=50304 下"总参 10M"基本不可行(d 得小到 64 才行)
      --size-by non-embedding  : 非 embedding 参数量(默认)。这才是"真正做计算的"那部分,
                                 也是缩放律里 N 的物理含义
    脚本默认按非 embedding 定阶梯, 并把 embedding 占比打在表里; 占比 >50% 会告警。

用法:
    python scripts/make_model_ladder.py                  # 14m(模板) → ~1B, 共 7 档
    python scripts/make_model_ladder.py --count 9        # 改档数
    python scripts/make_model_ladder.py --no-pin-template --min 70e6   # 不从模板起步
    python scripts/make_model_ladder.py --self-test      # 只校验参数量公式
    python scripts/make_model_ladder.py --d-granularity 128  # 贴 pythia 惯例(低端会变浅)
    python scripts/make_model_ladder.py --size-by total  # 换成总参数量口径
    python scripts/make_model_ladder.py --dry-run        # 只打印表, 不落盘
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time

LOG = logging.getLogger("ladder")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE_REPO = "EleutherAI/pythia-14m"
TEMPLATE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
# 官方源优先, 失败/慢自动切镜像
DEFAULT_ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")

# 长宽比律的锚点: pythia-14m 的 (d, L)。阶梯从它出发按 L ∝ d^p 外推。
ANCHOR_D, ANCHOR_L = 128, 6

# 公式自检用的参照: pythia 官方 9 档的 (d_model, n_layers, n_heads)。
# 用来验证 _params() 的系数(13Ld + 2d)没写错。
PYTHIA_REF = (
    ("pythia-14m", 128, 6, 4),
    ("pythia-70m", 512, 6, 8),
    ("pythia-160m", 768, 12, 12),
    ("pythia-410m", 1024, 24, 16),
    ("pythia-1b", 2048, 16, 16),
    ("pythia-1.4b", 2048, 24, 16),
    ("pythia-2.8b", 2560, 32, 32),
    ("pythia-6.9b", 4096, 32, 32),
    ("pythia-12b", 5120, 36, 40),
)


# ---------------------------------------------------------------------------
# 参数量公式
# ---------------------------------------------------------------------------
def params_nonemb(d: int, l: int) -> int:
    """非 embedding 参数量。12Ld² 是权重主体, 13Ld + 2d 是 bias/LayerNorm 的小项。"""
    return 12 * l * d * d + 13 * l * d + 2 * d


def params_embed(vocab: int, d: int) -> int:
    """输入 + 输出 embedding(不共享时是 2 份)。"""
    return 2 * vocab * d


def params_total(d: int, l: int, vocab: int) -> int:
    return params_nonemb(d, l) + params_embed(vocab, d)


def fmt_n(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))


# ---------------------------------------------------------------------------
# 尺寸搜索: 在 (d, L) 的离散网格上找最接近目标的解
# ---------------------------------------------------------------------------
def ideal_layers(d: int, p: float) -> float:
    """长宽比律给出的"理想层数": L = 6·(d/128)^p。"""
    return ANCHOR_L * (d / ANCHOR_D) ** p


def pick_heads(d: int, hd_min: int, hd_max: int) -> tuple[int, int]:
    """选 head_dim, 返回 (n_heads, head_dim)。

    规则: `head_dim = clamp(不小于 d/16 的最小 2 的幂, hd_min, hd_max)`。

    直觉是「**头数随 d 缓慢增长**」: d 要翻 16 倍才让 head_dim 翻一倍, 于是
    n_heads ≈ d/head_dim 随模型变大而变多。这条规则能复现 pythia 官方 9 档里的 8 档:

        14m(128)→hd32/h4    160m(768)→hd64/h12   410m(1024)→hd64/h16
        1b(2048)→hd128/h16  6.9b(4096)→hd128/h32  12b(5120)→hd128/h40

    (唯一明显不符的是 70m 用 hd64/h8, 以及 2.8b 用 hd80/h32 —— 后者不是 2 的幂。)
    若 d 不能被该 head_dim 整除, 就往下退一档 2 的幂。
    """
    want = hd_min
    while want < d / 16 and want < hd_max:
        want *= 2
    hd = want
    while hd >= hd_min:
        if d % hd == 0:
            return d // hd, hd
        hd //= 2
    raise SystemExit(f"d={d} 在 [{hd_min}, {hd_max}] 内没有能整除它的 2 的幂 head_dim")


def search_size(target: float, *, vocab: int, size_by: str, gran: int, p: float,
                lam: float, l_max: int, hd_min: int, hd_max: int,
                d_floor: int = 0, n_floor: float = 0.0) -> dict:
    """找使 |log(N/N_target)| + λ·|log(L/L_ideal)| 最小的 (d, L)。

    第一项想精确命中参数量, 第二项想把长宽比拉回定律 —— d 量化到 gran 的倍数后
    两者通常不能同时满足, λ 决定偏向谁。

    d_floor / n_floor 用于保证阶梯单调: 这一档的 d 必须**大于**上一档, 参数量也必须更大。
    """
    best = None
    for d in range(max(gran, d_floor), 8192 + 1, gran):
        l_ideal = ideal_layers(d, p)
        lo = max(2, int(l_ideal) - 8)
        hi = min(l_max, int(l_ideal) + 8)
        for l in range(lo, hi + 1):
            n = (params_nonemb(d, l) if size_by == "non-embedding"
                 else params_total(d, l, vocab))
            if n <= n_floor:
                continue
            score = abs(math.log(n / target)) + lam * abs(math.log(l / l_ideal))
            if best is None or score < best["score"]:
                n_heads, hd = pick_heads(d, hd_min, hd_max)
                best = {"score": score, "d": d, "l": l, "n": n, "n_heads": n_heads,
                        "head_dim": hd, "l_ideal": l_ideal}
    assert best is not None
    best["target"] = target
    best["nonemb"] = params_nonemb(best["d"], best["l"])
    best["total"] = params_total(best["d"], best["l"], vocab)
    best["emb"] = best["total"] - best["nonemb"]
    return best


def template_spec(base_cfg: dict, vocab: int, *, size_by: str, p: float) -> dict:
    """第 1 档 = 模板本身, 原样保留。

    不参与搜索的理由: 它是整条阶梯的锚点(长宽比律的标定点), 也是这批模型里**唯一
    有官方公开权重和训练曲线**的尺寸, 拿来当交叉验证的基准最有价值 —— 改掉它就失去
    了这个性质。
    """
    d = base_cfg["hidden_size"]
    l = base_cfg["num_hidden_layers"]
    n_heads = base_cfg["num_attention_heads"]
    nonemb = params_nonemb(d, l)
    total = nonemb + params_embed(vocab, d)
    l_ideal = ideal_layers(d, p)
    return {"score": 0.0, "d": d, "l": l, "n_heads": n_heads, "head_dim": d // n_heads,
            "l_ideal": l_ideal, "target": float(total if size_by == "total" else nonemb),
            "n": float(total if size_by == "total" else nonemb),
            "nonemb": nonemb, "total": total, "emb": total - nonemb, "pinned": True}


# ---------------------------------------------------------------------------
# 记录尺寸信息到 markdown
# ---------------------------------------------------------------------------
def write_markdown(path: str, *, specs: list[dict], a, targets: list[float],
                   template_repo: str, template_src: str, vocab: int,
                   total_flops_note: str = "") -> None:
    """把尺寸表和设计规则写成 markdown, 和 config 放一起, 便于随时查阅。"""
    def row(s: dict) -> str:
        share = s["emb"] / s["total"] * 100
        mark = " ⚠" if share > 50 else ""
        rung = "模板" if s.get("pinned") else f"{s['n'] / s['target'] - 1:+.1%}"
        return (f"| `{s['name']}` | {s['d']} | {s['l']} | {s['n_heads']} | {s['head_dim']} "
                f"| {4 * s['d']} | {fmt_n(s['nonemb'])} | {fmt_n(s['emb'])} | {fmt_n(s['total'])} "
                f"| {share:.1f}%{mark} | {s['l'] / s['l_ideal']:.3f} | {rung} |")

    L = [
        "# 模型尺寸阶梯",
        "",
        "> 本文件由 `scripts/make_model_ladder.py` 自动生成, **请勿手改** —— 改脚本重跑即可。",
        "",
        f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模板: `{template_repo}`(下载源 `{template_src}`)",
        f"- 档数: {len(specs)}(第 1 档为模板本身, 原样保留不搜索)",
        f"- 阶梯口径: **{a.size_by}** 参数量"
        + ("(非 embedding = 真正参与计算的那部分)" if a.size_by == "non-embedding" else ""),
        f"- 长宽比律: `L = {ANCHOR_L}·(d/{ANCHOR_D})^{a.depth_exponent:g}`"
        f",锚点 = 模板本身",
        f"- 离散约束: `d` 为 {a.d_granularity} 的倍数, "
        f"`head_dim = clamp(≥d/16 的最小 2 的幂, {a.head_dim_min}, {a.head_dim_max})`, "
        f"`ffw = 4·d`, `L` 为整数",
        f"- 词表: `vocab = {vocab}`(= 2 份 embedding 不共享, 合计 `2·{vocab}·d`)",
        "",
        "## 尺寸表",
        "",
        "| 目录名 | d_model | n_layers | n_heads | head_dim | ffw | N_非emb | embedding "
        "| N_总 | emb占比 | L/L_ideal | 目标偏差 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    L += [row(s) for s in specs]
    L += [
        "",
        f"- `N_非emb` = `12·L·d² + 13·L·d + 2·d`(真正参与计算的参数)",
        f"- `N_总` = `N_非emb + 2·vocab·d`(HF / pythia 命名用的是这个口径)",
        "- `目标偏差`: 实际尺寸与阶梯理论目标值的偏差; 第 1 档是模板, 不参与",
        "- `emb占比 > 50%` 标 ⚠: 该尺寸主要参数是词表而非计算层",
        "",
        "## 尺寸设计规则",
        "",
        "### 1. 骨架公式 —— 参数几乎只由 d 和 L 决定",
        "",
        "```",
        "N_非emb = 12·L·d² + 13·L·d + 2·d",
        "```",
        "",
        "`12Ld²` 是主体: 每层 `4d²`(注意力 QKV+O) + `8d²`(MLP, 因为 `ffw = 4d` 时是 `2·4d·d`)。",
        "`13Ld + 2d` 是 bias 和 LayerNorm 的小项, 占比不到 1%。",
        "",
        "**`d` 是平方项, `L` 是一次项 —— `d` 翻倍 = 参数 ×4, `L` 翻倍 = 参数 ×2。**",
        "所以放大模型主要靠放大 `d`。",
        "",
        "### 2. 长宽比律 —— 怎么从 14M 推到 1B",
        "",
        "`d` 和 `L` 必须一起长, 且 `d` 长得更快。用 pythia 家族两端标定(非 embedding 口径):",
        "",
        "```",
        "14m: N=1.19M,  d=128,  L=6",
        "12b: N=11.33B, d=5120, L=36",
        "→ d ∝ N^0.403 ,  L ∝ N^0.196    取整即  d ∝ N^0.4, L ∝ N^0.2",
        "```",
        "",
        "等价形式就是脚本用的约束:",
        "",
        "```",
        "L ∝ d^0.5     即   L = 6·(d/128)^0.5",
        "```",
        "",
        "(自洽性: `0.4×2 + 0.2 = 1.0` ✓, 因为 `N ∝ L·d²`。)",
        "",
        "**所以「从 14M 放大到 1B」的算法只有三步** —— 比如目标从非 emb 1.19M 到 1018M(涨 856×):",
        "",
        "| 步骤 | 计算 | 结果 |",
        "|---|---|---|",
        "| ① 参数涨几倍 | 1018M / 1.19M | 856× |",
        "| ② `d` 涨多少 | 856^0.4 | 15× → d: 128 → 1920 |",
        "| ③ `L` 涨多少 | 856^0.2 | 3.8× → L: 6 → 23 |",
        "| ④ 验算 | 15² × 3.8 | 855 ✓ |",
        "",
        "缩小就反过来: 把 `d` 除以 15、`L` 除以 3.8。**不是拍脑袋选尺寸, 而是把 `log(N)`",
        "按 4:2 分给 `log(d)` 和 `log(L)`。**",
        "",
        "### 3. 硬约束与「算得出但拿不到」",
        "",
        "```",
        f"d 必须是 {a.d_granularity} 的倍数           → tensor core 对齐",
        f"head_dim = clamp(≥d/16 的最小 2 的幂, {a.head_dim_min}, {a.head_dim_max})"
        f"   → 「头数随 d 缓慢增长」",
        "ffw = 4·d                       → pythia / Chinchilla 一致惯例",
        "L 必须是整数, 且 d 必须逐档递增",
        "```",
        "",
        "`head_dim` 那条规则能复现 pythia 官方 9 档里的 8 档(14m→hd32/h4、160m→hd64/h12、",
        "410m→hd64/h16、1b→hd128/h16、6.9b→hd128/h32、12b→hd128/h40)。含义是: d 要翻 16 倍",
        "才让 head_dim 翻一倍, 于是头数 `n_heads ≈ d/head_dim` 随模型变大而变多。",
        "",
        "`d` 被量化(且必须逐档递增)后, **不可能同时精确命中目标参数量和理想长宽比**。",
        "脚本用加权评分折中:",
        "",
        "```",
        "minimize  |log(N/N_target)| + λ·|log(L/L_ideal)|        λ = --aspect-penalty",
        "```",
        "",
        "表里 `L/L_ideal` 和 `目标偏差` 两列就是把这两个偏差摊开。若把 `--d-granularity`",
        "调回 128(pythia 惯例), 低端 d 会一跳 2 倍、`L` 被迫压扁成浅模型, 不建议。",
        "",
        "### 4. ⚠️ embedding 陷阱: 「10M 模型」到底指什么",
        "",
        f"`vocab={vocab}` 时, embedding 是 `2·{vocab}·d = {2 * vocab}·d` 个参数, **与 `L` 无关**。",
        "`d=256` 时它就是 25.8M —— 比整个「10M 模型」还大。所以:",
        "",
        "- `--size-by non-embedding`(默认): 按非 embedding 定尺寸。这才是真正做计算的部分,",
        "  也是缩放律里 `N` 的物理含义",
        "- `--size-by total`: 按总参数量(也就是 pythia/HF 命名的口径)。小端会退化成",
        "  「小 transformer + 大查找表」",
        "",
        "**目录名用的是总参数量口径**(和 pythia 官方命名一致), 所以会看到",
        "「名义 14m、非 emb 只有 1.2M」—— 这不矛盾, 是两个口径。",
        "",
        "### 5. 超参必须随尺寸变(否则训不起来)",
        "",
        "- **max LR ∝ 1/d**(宽的模型要更小的 LR)",
        "- **batch size(按 token 计)随模型增大而增大**",
        "- **AdamW β2**: 小 batch 下用 0.99/0.999, **不要用 0.95**",
        "- **warmup tokens ≈ N**",
        "- ⚠️ **初始化**: HF 的 `GPTNeoXPreTrainedModel._init_weights` 对**所有尺寸**都用固定的",
        "  `initializer_range=0.02`(`normal_(0, 0.02)`), **没有** EleutherAI 原实现里 `1/√(2L)`",
        "  的残差输出缩放。含义: ① 大 `L` 档若 loss 一开头就炸, 先怀疑 init 而不是自己写错了;",
        "  ② 在 HF 里 from-scratch 训出来的模型, 与 pythia 官方权重**不是同一套 init 方案**,",
        "  交叉验证时要留这个偏差",
        "",
        "## 复现",
        "",
        "```bash",
        f"python scripts/make_model_ladder.py --count {a.count} --max {a.max:.0e} "
        f"--size-by {a.size_by} --d-granularity {a.d_granularity} "
        f"--depth-exponent {a.depth_exponent:g} --aspect-penalty {a.aspect_penalty:g}",
        "```",
        "",
        "生成的每个目录只有 `config.json` + tokenizer(**无权重**), 供 LLaMA-Factory 的",
        "`train_from_scratch: true` 从零初始化。tokenizer 默认是软链接指向模板 —— ",
        "**新 clone 后软链接会悬空, 必须先跑一次本脚本把模板下下来**(或用 `--copy-tokenizer`)。",
        "",
    ]
    if total_flops_note:
        L += [total_flops_note, ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------------------
# 模板下载(官方源 → 镜像, 都打日志)
# ---------------------------------------------------------------------------
_FETCH = (
    "import os,sys;"
    "os.environ['HF_ENDPOINT']=sys.argv[1];"
    "from huggingface_hub import snapshot_download;"
    "snapshot_download(repo_id=sys.argv[2],local_dir=sys.argv[3],"
    "allow_patterns=sys.argv[4].split(','))"
)


def fetch_template(repo: str, dest: str, files: tuple[str, ...], endpoints: tuple[str, ...],
                   force: bool) -> str:
    """把模板文件下到 dest。返回实际生效的源。"""
    if all(os.path.exists(os.path.join(dest, f)) for f in files) and not force:
        LOG.info("模板已存在, 跳过下载: %s", dest)
        return "cache"

    env = dict(os.environ)
    # token 只从环境透传, 不落盘(HF_TOKEN 由调用方 export)
    for ep in endpoints:
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-c", _FETCH, ep, repo, dest, ",".join(files)],
            capture_output=True, text=True, env=env)
        dt = time.time() - t0
        if r.returncode == 0:
            LOG.info("模板下载成功: repo=%s 源=%s 用时=%.1fs -> %s", repo, ep, dt, dest)
            return ep
        last = (r.stderr or r.stdout or "").strip().splitlines()
        LOG.warning("源 %s 失败(%.1fs): %s", ep, dt, last[-1] if last else "未知错误")
    raise SystemExit(f"所有源都下载失败: {endpoints}")


# ---------------------------------------------------------------------------
# 写模型目录
# ---------------------------------------------------------------------------
def write_model(out_dir: str, base_cfg: dict, spec: dict, *, seq_len: int | None,
                dtype: str | None, copy_tokenizer: bool, template_dir: str) -> None:
    """写 config.json + tokenizer。只覆盖 4 个尺寸字段, 其余全部沿用模板。"""
    os.makedirs(out_dir, exist_ok=True)

    cfg = dict(base_cfg)
    cfg["hidden_size"] = spec["d"]
    cfg["num_hidden_layers"] = spec["l"]
    cfg["num_attention_heads"] = spec["n_heads"]
    cfg["intermediate_size"] = 4 * spec["d"]
    if seq_len is not None:
        cfg["max_position_embeddings"] = seq_len
    if dtype is not None:
        cfg["torch_dtype"] = dtype

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    # tokenizer.json 有 2MB 且 7 份完全相同 —— 默认用软链接省 ~15MB(--copy-tokenizer 可改真拷贝)
    for name in TEMPLATE_FILES[1:]:
        src = os.path.join(template_dir, name)
        dst = os.path.join(out_dir, name)
        if os.path.lexists(dst):
            os.remove(dst)
        if copy_tokenizer:
            shutil.copy(src, dst)
        else:
            os.symlink(os.path.relpath(src, out_dir), dst)


# ---------------------------------------------------------------------------
# 用 transformers 在 meta device 上实测参数量(校验公式, 不占显存)
# ---------------------------------------------------------------------------
_MODEL_CLASSES = {
    "gpt_neox": ("transformers.models.gpt_neox.modeling_gpt_neox", "GPTNeoXForCausalLM"),
}


def measure_with_transformers(cfg_dir: str) -> tuple[tuple[int, int] | None, str | None]:
    """返回 ((总参, embedding 参), 错误信息)。"""
    try:
        import torch
        from transformers import AutoConfig
    except Exception as e:  # noqa: BLE001
        return None, f"transformers/torch 不可用: {e}"

    cfg = AutoConfig.from_pretrained(cfg_dir)
    target = _MODEL_CLASSES.get(cfg.model_type)
    if target is None:
        return None, f"未登记 model_type={cfg.model_type}"

    # 注意: 模型类必须在进 meta 上下文**之前** import 好。
    # torch 2.6 + transformers 4.51 里, 在 meta 上下文中触发 lazy import 会在
    # flex_attention 的初始化里对 meta tensor 调 .item() 而崩溃。
    cls = getattr(importlib.import_module(target[0]), target[1])
    try:
        with torch.device("meta"):
            model = cls(cfg)
        total = sum(p.numel() for p in model.parameters())
        emb = sum(p.numel() for n, p in model.named_parameters()
                  if n.startswith(("gpt_neox.embed_in", "embed_out")))
        return (total, emb), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def self_test(vocab: int) -> bool:
    """把公式算的参数量与 transformers 实建的模型对一遍(pythia 官方 9 档)。"""
    LOG.info("=" * 100)
    LOG.info("公式自检: 12Ld² + 13Ld + 2d (+2Vd)  对 pythia 官方尺寸")
    LOG.info("%-13s %6s %4s %5s | %14s %14s %8s", "名称", "d", "L", "heads",
             "公式总参", "实建总参", "误差")
    ok = True
    for name, d, l, h in PYTHIA_REF:
        cfg_dir = os.path.join("/tmp", f"_selftest_{name}")
        os.makedirs(cfg_dir, exist_ok=True)
        with open(os.path.join(cfg_dir, "config.json"), "w") as f:
            json.dump({"architectures": ["GPTNeoXForCausalLM"], "model_type": "gpt_neox",
                       "hidden_size": d, "num_hidden_layers": l, "num_attention_heads": h,
                       "intermediate_size": 4 * d, "max_position_embeddings": 2048,
                       "vocab_size": vocab, "rotary_pct": 0.25, "rotary_emb_base": 10000,
                       "layer_norm_eps": 1e-5, "hidden_act": "gelu",
                       "use_parallel_residual": True, "attention_bias": True,
                       "tie_word_embeddings": False}, f)
        formula = params_total(d, l, vocab)
        got, err = measure_with_transformers(cfg_dir)
        if got is None:
            LOG.warning("%-13s 实建失败: %s", name, err)
            ok = False
            continue
        real_total, real_emb = got
        diff = real_total - formula
        ok &= diff == 0
        LOG.info("%-13s %6d %4d %5d | %14s %14s %8d", name, d, l, h,
                 f"{formula:,}", f"{real_total:,}", diff)
    LOG.info("自检结果: %s", "全部一致 ✅" if ok else "存在偏差 ❌")
    LOG.info("=" * 100)
    return ok


# ---------------------------------------------------------------------------
def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"make_model_ladder_{time.strftime('%Y%m%d_%H%M%S')}.log")
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        h.setFormatter(fmt)
        LOG.addHandler(h)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="从 pythia-14m 模板派生尺寸阶梯(只生成 config + tokenizer, 不含权重)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法:")[-1] if __doc__ else None)
    ap.add_argument("--root", default=ROOT, help="项目根目录")
    ap.add_argument("--template-repo", default=TEMPLATE_REPO, help="HF 模板仓库")
    ap.add_argument("--endpoints", default=",".join(DEFAULT_ENDPOINTS),
                    help="下载源, 按顺序尝试")
    ap.add_argument("--force", action="store_true", help="强制重新下载模板")

    # 阶梯
    ap.add_argument("--min", type=float, default=10e6, help="最小尺寸(参数量), 仅 --no-pin-template 时生效")
    ap.add_argument("--max", type=float, default=1e9, help="最大尺寸(参数量)")
    ap.add_argument("--count", type=int, default=7, help="档数(含最小档)")
    ap.add_argument("--pin-template", dest="pin_template", action="store_true", default=True,
                    help="第 1 档 = 模板本身原样保留(默认开)")
    ap.add_argument("--no-pin-template", dest="pin_template", action="store_false",
                    help="第 1 档也参与搜索, 阶梯从 --min 起")
    ap.add_argument("--ratio", type=float, default=0.0,
                    help="相邻档位比值; 给了就忽略 --min/--max/--count 的比值推导")
    ap.add_argument("--size-by", choices=["non-embedding", "total"], default="non-embedding",
                    help="阶梯按哪个口径定尺寸(默认非 embedding, 见文末说明)")

    # 尺寸设计旋钮
    ap.add_argument("--d-granularity", type=int, default=64,
                    help="d_model 的量化步长。64(默认)让低端也能逐档增长; "
                         "128 更贴 pythia 惯例但低端 d 会跳档、L 被迫压扁")
    ap.add_argument("--depth-exponent", type=float, default=0.5,
                    help="长宽比律 L ∝ d^p 的 p(0.5 = 默认)")
    ap.add_argument("--aspect-penalty", type=float, default=0.5,
                    help="长宽比偏差的权重 λ; 调大 → 更守定律但更偏参数量目标")
    ap.add_argument("--max-layers", type=int, default=96)
    ap.add_argument("--head-dim-min", type=int, default=32,
                    help="head_dim 下限(2 的幂)")
    ap.add_argument("--head-dim-max", type=int, default=128,
                    help="head_dim 上限(2 的幂)")

    # 其它
    ap.add_argument("--seq-len", type=int, default=None, help="覆盖 max_position_embeddings")
    ap.add_argument("--dtype", default=None, help="覆盖 torch_dtype(默认沿用模板)")
    ap.add_argument("--copy-tokenizer", action="store_true",
                    help="真拷贝 tokenizer(默认软链接, 省 ~15MB)")
    ap.add_argument("--no-md", action="store_true",
                    help="不写 models/configs/README.md 尺寸记录")
    ap.add_argument("--verify", dest="verify", action="store_true", default=True)
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="跳过 transformers 实建校验")
    ap.add_argument("--self-test", action="store_true", help="只跑参数量公式自检")
    ap.add_argument("--dry-run", action="store_true", help="只打印表, 不落盘")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    log_path = setup_logging(os.path.join(root, "logs"))
    LOG.info("日志: %s", log_path)
    LOG.info("启动参数: %s", vars(a))

    template_dir = os.path.join(root, "models", "templates", a.template_repo.split("/")[-1])
    out_root = os.path.join(root, "models", "configs")

    if a.self_test:
        self_test(50304)
        return

    # ---- 1. 模板 ----
    endpoints = tuple(x.strip() for x in a.endpoints.split(",") if x.strip())
    used = fetch_template(a.template_repo, template_dir, TEMPLATE_FILES, endpoints, a.force)
    with open(os.path.join(template_dir, "config.json")) as f:
        base_cfg = json.load(f)
    vocab = base_cfg["vocab_size"]
    LOG.info("模板: %s  源=%s  vocab=%d  d=%d  L=%d  heads=%d  ffw=%d  seq=%d  tie=%s",
             a.template_repo, used, vocab, base_cfg["hidden_size"],
             base_cfg["num_hidden_layers"], base_cfg["num_attention_heads"],
             base_cfg["intermediate_size"], base_cfg["max_position_embeddings"],
             base_cfg.get("tie_word_embeddings"))

    # ---- 2. 造阶梯 ----
    # 第 1 档默认就是模板自己(不搜索), 从它往上几何递增到 --max。
    if a.pin_template:
        lo = (params_total(base_cfg["hidden_size"], base_cfg["num_hidden_layers"], vocab)
              if a.size_by == "total"
              else params_nonemb(base_cfg["hidden_size"], base_cfg["num_hidden_layers"]))
    else:
        lo = a.min
    if a.ratio > 0:
        targets = [lo * a.ratio ** i for i in range(a.count)]
    else:
        r = (a.max / lo) ** (1.0 / (a.count - 1)) if a.count > 1 else 1.0
        targets = [lo * r ** i for i in range(a.count)]
    LOG.info("阶梯口径=%s  第1档锚点=%.3e  公比=%.3f", a.size_by, lo, r if a.count > 1 else 1.0)
    LOG.info("目标: %s", " → ".join(fmt_n(t) for t in targets))
    LOG.info("长宽比律: L = %d·(d/%d)^%.2f  (锚点 = 模板 %s)",
             ANCHOR_L, ANCHOR_D, a.depth_exponent, a.template_repo.split("/")[-1])

    specs = []
    for i, t in enumerate(targets):
        if i == 0 and a.pin_template:
            s = template_spec(base_cfg, vocab, size_by=a.size_by, p=a.depth_exponent)
            LOG.info("第 1 档 = 模板本身, 原样保留(d=%d L=%d heads=%d)",
                     s["d"], s["l"], s["n_heads"])
        else:
            prev = specs[-1]
            s = search_size(t, vocab=vocab, size_by=a.size_by, gran=a.d_granularity,
                            p=a.depth_exponent, lam=a.aspect_penalty, l_max=a.max_layers,
                            hd_min=a.head_dim_min, hd_max=a.head_dim_max,
                            d_floor=prev["d"] + a.d_granularity, n_floor=prev["n"])
        s["name"] = ladder_name(s["total"])
        specs.append(s)

    # ---- 3. 打印表 ----
    LOG.info("=" * 112)
    LOG.info("尺寸表  (N_非emb = 12Ld² + 13Ld + 2d;  embedding = 2·%d·d)", vocab)
    LOG.info("%-12s %6s %4s %5s %8s %6s | %9s %9s %9s %8s | %7s %8s",
             "目录名", "d", "L", "heads", "head_dim", "ffw",
             "N_非emb", "emb", "N_总", "emb占比", "L/L_ideal", "目标偏差")
    for s in specs:
        emb_share = s["emb"] / s["total"]
        flag = " ⚠" if emb_share > 0.5 else ""
        dev = "模板" if s.get("pinned") else f"{(s['n'] / s['target'] - 1) * 100:+.1f}%"
        LOG.info("%-12s %6d %4d %5d %8d %6d | %9s %9s %9s %7.1f%%%s | %7.3f %8s",
                 s["name"], s["d"], s["l"], s["n_heads"], s["head_dim"], 4 * s["d"],
                 fmt_n(s["nonemb"]), fmt_n(s["emb"]), fmt_n(s["total"]),
                 emb_share * 100, flag, s["l"] / s["l_ideal"], dev)
    LOG.info("=" * 112)
    LOG.info("注: 目录名用**总参数量**口径(与 pythia 官方命名一致); N_非emb 才是计算层参数")
    if a.size_by == "total":
        LOG.warning("按总参数量定阶梯: vocab=%d 的 embedding 会吃掉小端大部分参数, "
                    "小模型实际是「小 transformer + 大查找表」", vocab)
    for s in specs:
        if s["emb"] / s["total"] > 0.5:
            LOG.warning("%s: embedding 占 %.0f%% —— 主要参数是词表, 不是计算层",
                        s["name"], s["emb"] / s["total"] * 100)

    if a.dry_run:
        LOG.info("[dry-run] 未落盘。")
        return

    # ---- 4. 落盘 ----
    total_units = 0
    for s in specs:
        out_dir = os.path.join(out_root, s["name"])
        write_model(out_dir, base_cfg, s, seq_len=a.seq_len, dtype=a.dtype,
                    copy_tokenizer=a.copy_tokenizer, template_dir=template_dir)
        total_units += 1
        LOG.info("已写出: %s  (d=%d L=%d heads=%d ffw=%d)",
                 out_dir, s["d"], s["l"], s["n_heads"], 4 * s["d"])

    # ---- 5. 实建校验 ----
    if a.verify:
        LOG.info("-" * 100)
        LOG.info("实建校验(transformers 在 meta device 上建模型数参数, 不占显存)")
        LOG.info("%-12s %14s %14s %10s", "名称", "公式总参", "实建总参", "误差")
        bad = 0
        for s in specs:
            got, err = measure_with_transformers(os.path.join(out_root, s["name"]))
            if got is None:
                LOG.warning("%-12s 实建失败: %s", s["name"], err)
                bad += 1
                continue
            real_total, real_emb = got
            diff = real_total - s["total"]
            bad += diff != 0
            LOG.info("%-12s %14s %14s %10d", s["name"], f"{s['total']:,}",
                     f"{real_total:,}", diff)
        LOG.info("校验: %s", "全部一致 ✅" if bad == 0 else f"{bad} 个不一致 ❌")

    # ---- 6. 记尺寸信息到 markdown ----
    if not a.no_md:
        md_path = os.path.join(out_root, "README.md")
        write_markdown(md_path, specs=specs, a=a, targets=targets,
                       template_repo=a.template_repo, template_src=used, vocab=vocab)
        LOG.info("尺寸信息已记录: %s", md_path)

    LOG.info("完成: %d 个模型目录 -> %s", total_units, out_root)
    LOG.info("tokenizer: %s", "真拷贝" if a.copy_tokenizer else "软链接(指向模板)")


def ladder_name(total_params: float) -> str:
    """按 pythia 的命名格式给一档起名: 14m / 215m / 1.2b。

    口径 = **总参数量**, 和 pythia 官方一致(官方 pythia-14m 总参正是 14.07M, 所以
    模板那一档会自然得到 "pythia-14m", 与官方目录名对得上)。
    """
    m = total_params / 1e6
    if m >= 1000:
        return f"pythia-{round(m / 1000, 1):g}b"
    return f"pythia-{round(m)}m"


if __name__ == "__main__":
    main()
