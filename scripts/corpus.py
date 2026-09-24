#!/usr/bin/env python3
"""fineweb parquet 语料 -> LLaMA-Factory 的 pt 数据集。

只负责"数据这一侧"的三件事, 不碰模型也不碰训练:

  1. 盘点(inventory): 每个 parquet 有多少 token。只读 parquet footer 里的
     token_count 列, 不读正文 —— 282G 语料盘点一次只要几秒。
  2. 规划(plan): 按 token 预算挑 shard。用 run 名做种子选一段连续切片,
     同一个 run 名永远挑同一批数据(可复现), 不同 run 挑不同切片(避免整个
     网格共享同一批数据带来的顺序混淆)。
  3. 落地(materialize): 建出 LLaMA-Factory 认的目录结构

         <data_dir>/dataset_info.json
         <data_dir>/raw/shard_0000.parquet -> 语料里的某个 parquet(软链接)

     用软链接而不是拷贝: 语料只有一份, 150 个 parquet 一共 282G,
     每个 run 都复制一份会把配额吃光。

为什么是 `formatting: alpaca` + `columns: {prompt: text}`:
    LLaMA-Factory 的 pt(预训练)阶段最终读的是 `_prompt`(`pretrain.py:30`),
    而 `_prompt` 由 `columns.prompt` 指定的那一列填充。fineweb 的正文列叫
    `text`, 所以必须映射过去, 不能直接吃默认的 `instruction`。

token 口径:
    parquet 自带的 `token_count` 是 fineweb 自家 tokenizer(gpt2 系)的口径,
    与我们用的 Pythia tokenizer 不同。实测抽样 3000 条约 1.0033 倍偏差
    (≤0.4%), 所以直接用它做预算, 不需要为了数 token 把语料跑一遍。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq

LOG = logging.getLogger("corpus")

# 盘点结果缓存到 /tmp —— 只是个缓存, 丢了重建即可(几秒)
CACHE_DIR = Path("/tmp/scalinglaw")
# 实测: parquet 自报的 token_count 相对 Pythia tokenizer 的偏差
TOKEN_COUNT_SKEW = 1.0033


@dataclass
class Shard:
    """一个 parquet 分片。"""

    path: str
    rows: int
    tokens: int  # parquet 自报口径(已足够准, 见 TOKEN_COUNT_SKEW)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def inventory(corpus_dir: str | os.PathLike, use_cache: bool = True) -> list[Shard]:
    """盘点语料目录下所有 parquet, 返回按文件名排序的分片列表。

    只读 token_count 一列(每文件约 8MB), 不读正文 —— 这是整个流程能秒级
    给出 token 预算的关键。
    """
    corpus_dir = Path(corpus_dir)
    files = sorted(corpus_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{corpus_dir} 下没有 *.parquet")

    key = hashlib.sha1(str(corpus_dir.resolve()).encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"corpus_inventory_{key}.json"
    # 缓存要能识别"语料目录内容变了"(比如又下了几个文件)
    stamp = f"{len(files)}:{files[0].stat().st_size}:{files[-1].stat().st_size}:{files[-1].stat().st_mtime_ns}"

    if use_cache and cache.exists():
        try:
            blob = json.loads(cache.read_text())
            if blob.get("stamp") == stamp:
                shards = [Shard(**s) for s in blob["shards"]]
                LOG.info("盘点命中缓存: %d 个分片, 共 %.2fB tokens", len(shards), sum(s.tokens for s in shards) / 1e9)
                return shards
        except (json.JSONDecodeError, TypeError, KeyError):
            LOG.warning("盘点缓存损坏, 重新盘点")

    LOG.info("开始盘点 %d 个 parquet(只读 token_count 列)...", len(files))
    t0 = time.time()
    shards: list[Shard] = []
    for i, f in enumerate(files, 1):
        tbl = pq.read_table(f, columns=["token_count"])
        shards.append(
            Shard(
                path=str(f),
                rows=tbl.num_rows,
                tokens=int(sum(tbl.column("token_count").to_pylist())),
            )
        )
        if i % 25 == 0 or i == len(files):
            done = sum(s.tokens for s in shards)
            LOG.info(
                "  盘点进度 %d/%d (%.0f%%), 已数 %.2fB tokens, 用时 %.1fs",
                i, len(files), 100 * i / len(files), done / 1e9, time.time() - t0,
            )

    total = sum(s.tokens for s in shards)
    LOG.info(
        "盘点完成: %d 个分片, %.2fB tokens(自报口径), 平均每片 %.3fB, 用时 %.1fs",
        len(shards), total / 1e9, total / len(shards) / 1e9, time.time() - t0,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"stamp": stamp, "shards": [asdict(s) for s in shards]}, indent=1))
    return shards


def plan(shards: list[Shard], tokens_needed: int, seed_name: str, offset: int | None = None) -> tuple[list[Shard], int]:
    """挑一段连续分片, 保证其 token 总量 >= tokens_needed。

    连续(而不是随机散落)是为了让每个 run 的顺序读友好; 起点由 seed_name 决定,
    所以不同 run 看到不同数据切片, 但同一 run 重跑可复现。

    返回 (挑中的分片, 起点下标)。
    """
    if tokens_needed <= 0:
        raise ValueError(f"tokens_needed 必须为正, 得到 {tokens_needed}")

    picked: list[Shard] = []
    total = 0
    # 从起点开始往后取, 不够就绕回开头继续取(环形)
    if offset is None:
        n_files = len(shards)
        # 留出余量, 避免起点太靠后导致可用分片不足
        span = sum(s.tokens for s in shards)
        avg = span / n_files
        need_n = max(1, int(tokens_needed / avg) + 1)
        room = max(1, n_files - need_n)
        offset = int(hashlib.sha1(seed_name.encode()).hexdigest()[:8], 16) % room

    i = offset
    seen = 0
    while total < tokens_needed:
        if seen >= len(shards):
            raise ValueError(
                f"语料总量不足: 需要 {tokens_needed/1e9:.2f}B, 全部 {len(shards)} 片只有 "
                f"{sum(s.tokens for s in shards)/1e9:.2f}B"
            )
        picked.append(shards[i % len(shards)])
        total += shards[i % len(shards)].tokens
        i += 1
        seen += 1

    return picked, offset


def materialize(data_dir: str | os.PathLike, picked: list[Shard], dataset_name: str) -> dict:
    """把挑中的分片落成 LLaMA-Factory 的数据集目录。

    产出:
        <data_dir>/dataset_info.json   内容是一个只含该数据集的 dict
        <data_dir>/raw/shard_NNNN.parquet -> 原 parquet 的软链接

    返回 dataset_info.json 里那一条配置(便于日志和复现)。
    """
    data_dir = Path(data_dir)
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # 清掉上一轮的软链接, 否则换预算时旧分片会残留 -> 训练看到的数据比预期多
    for old in raw.iterdir():
        if old.is_symlink() or old.is_file():
            old.unlink()

    for i, shard in enumerate(picked):
        link = raw / f"shard_{i:04d}.parquet"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(os.path.abspath(shard.path))

    entry = {
        "file_name": "raw",  # 目录 -> LLaMA-Factory 会 glob 里面所有文件
        "formatting": "alpaca",
        "columns": {"prompt": "text"},  # pt 阶段读 _prompt, 必须映射到正文列
    }
    info = {dataset_name: entry}
    (data_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")

    LOG.info(
        "数据集已落地: %s (%d 个分片 -> raw/, 共 %.2fB tokens)",
        data_dir, len(picked), sum(s.tokens for s in picked) / 1e9,
    )
    return entry


def prepare(
    corpus_dir: str | os.PathLike,
    data_dir: str | os.PathLike,
    tokens_needed: int,
    dataset_name: str,
    seed_name: str,
    offset: int | None = None,
) -> dict:
    """盘点 + 规划 + 落地, 一步到位。返回本次实际可用的 token 数等信息。"""
    shards = inventory(corpus_dir)
    picked, start = plan(shards, tokens_needed, seed_name, offset)
    entry = materialize(data_dir, picked, dataset_name)
    return {
        "dataset_name": dataset_name,
        "entry": entry,
        "shards": len(picked),
        "shard_offset": start,
        "shard_names": [s.name for s in picked],
        "available_tokens": sum(s.tokens for s in picked),
    }


if __name__ == "__main__":
    # 单独跑: 盘点一下语料, 确认可用
    import argparse

    ap = argparse.ArgumentParser(description="盘点 fineweb parquet 语料")
    ap.add_argument("--corpus", default="/volume/data/hjiang02/open_source/datasets/fineweb/sample/100BT")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sh = inventory(a.corpus, use_cache=not a.no_cache)
    tot = sum(s.tokens for s in sh)
    print(f"\n分片数   : {len(sh)}")
    print(f"总 tokens: {tot/1e9:.2f}B (自报口径, 校正 {TOKEN_COUNT_SKEW} 后 {tot*TOKEN_COUNT_SKEW/1e9:.2f}B)")
    print(f"总行数   : {sum(s.rows for s in sh):,}")
    print(f"\n前 3 片: {', '.join(f'{s.name}={s.tokens/1e9:.3f}B' for s in sh[:3])}")
    print(f"后 3 片: {', '.join(f'{s.name}={s.tokens/1e9:.3f}B' for s in sh[-3:])}")
