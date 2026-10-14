#!/usr/bin/env python3
"""按 token 预算选择 parquet 分片，生成数据配置和软链接。"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

LOG = logging.getLogger("corpus")

CACHE_DIR = Path("/tmp/scalinglaw")

TOKEN_COUNT_SKEW = 1.0033


@dataclass
class Shard:
    """一个 parquet 分片。"""

    path: str
    rows: int
    tokens: int

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def inventory(corpus_dir, use_cache: bool = True) -> list[Shard]:
    """读取 token_count 列，缓存每个分片的行数和 token 数。"""
    files = sorted(Path(corpus_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{corpus_dir} 下没有 parquet")
    key = hashlib.sha1(str(Path(corpus_dir).resolve()).encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"corpus_inventory_{key}.json"
    stamp = [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files]
    if use_cache and cache.exists():
        try:
            saved = json.loads(cache.read_text())
            if saved["stamp"] == stamp:
                return [Shard(**s) for s in saved["shards"]]
        except (ValueError, KeyError, TypeError):
            LOG.warning("缓存无效，重新盘点")
    shards = []
    for index, path in enumerate(files, 1):
        table = pq.read_table(path, columns=["token_count"])
        shards.append(
            Shard(str(path.resolve()), table.num_rows, int(pc.sum(table["token_count"]).as_py()))
        )
        if index % 25 == 0 or index == len(files):
            LOG.info("盘点 %d/%d 个分片", index, len(files))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"stamp": stamp, "shards": [asdict(s) for s in shards]}))
    return shards


def plan(shards: list[Shard], tokens_needed: int, seed_name: str, offset=None):
    """按种子选择连续分片，保持同一实验的数据选择可复现。"""
    available = sum(s.tokens for s in shards)
    if tokens_needed <= 0 or tokens_needed > available:
        raise ValueError(f"token 预算必须在 (0, {available}] 内")
    if offset is None:
        needed = max(1, int(tokens_needed / (available / len(shards))) + 1)
        room = max(1, len(shards) - needed)
        offset = int(hashlib.sha1(seed_name.encode()).hexdigest()[:8], 16) % room
    picked, total = [], 0
    for i in range(offset, offset + len(shards)):
        shard = shards[i % len(shards)]
        picked.append(shard)
        total += shard.tokens
        if total >= tokens_needed:
            return picked, offset
    raise ValueError("语料不足")


def materialize(data_dir: str | os.PathLike, picked: list[Shard], dataset_name: str) -> dict:
    """生成数据配置和软链接，相同切片可重复执行。"""
    data_dir = Path(data_dir)
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    expected = {f"shard_{i:04d}.parquet": str(Path(s.path).resolve()) for i, s in enumerate(picked)}
    for old in raw.iterdir():
        if (
            old.name not in expected
            or not old.is_symlink()
            or str(old.resolve()) != expected[old.name]
        ):
            raise FileExistsError(f"{raw} 已有不同切片或其他文件，请使用新的 data-dir")

    for i, shard in enumerate(picked):
        link = raw / f"shard_{i:04d}.parquet"
        if not link.is_symlink():
            link.symlink_to(os.path.abspath(shard.path))

    entry = {
        "file_name": "raw",
        "formatting": "alpaca",
        "columns": {"prompt": "text"},
    }
    info = {dataset_name: entry}
    (data_dir / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n"
    )

    LOG.info(
        "数据集已落地: %s (%d 个分片 -> raw/, 共 %.2fB tokens)",
        data_dir,
        len(picked),
        sum(s.tokens for s in picked) / 1e9,
    )
    return entry


def prepare(
    corpus_dir: str | os.PathLike,
    data_dir: str | os.PathLike,
    tokens_needed: int,
    dataset_name: str,
    seed_name: str,
    offset: int | None = None,
    use_cache: bool = True,
) -> dict:
    """盘点 + 规划 + 落地, 一步到位。返回本次实际可用的 token 数等信息。"""
    shards = inventory(corpus_dir, use_cache=use_cache)
    picked, start = plan(shards, tokens_needed, seed_name, offset)
    entry = materialize(data_dir, picked, dataset_name)
    metadata = {
        "dataset_name": dataset_name,
        "entry": entry,
        "shards": len(picked),
        "shard_offset": start,
        "shard_names": [s.name for s in picked],
        "available_tokens": sum(s.tokens for s in picked),
    }
    (Path(data_dir) / "corpus.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    return metadata


def main() -> None:
    """独立盘点或准备数据；预算 JSON 是文件接口，不依赖训练模块。"""
    import argparse

    ap = argparse.ArgumentParser(description="盘点语料，或按 token 预算准备预训练数据")
    ap.add_argument(
        "--corpus", default="/volume/data/hjiang02/open_source/datasets/fineweb/sample/100BT"
    )
    ap.add_argument("--no-cache", action="store_true")
    budget = ap.add_mutually_exclusive_group()
    budget.add_argument(
        "--budget-file", help="读取预算 JSON；用 - 从标准输入读取（train.sh 使用此接口）"
    )
    budget.add_argument("--tokens", type=float, help="独立准备数据时需要的 token 数")
    ap.add_argument("--data-dir", help="准备好的数据目录；预算模式默认取 JSON 的 data_dir")
    ap.add_argument("--dataset", help="数据集名；预算模式默认取 JSON 的 dataset_name")
    ap.add_argument("--seed-name", help="可复现切片的种子名；预算模式默认取 run_name")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.budget_file or a.tokens is not None:
        spec = {}
        if a.budget_file:
            spec = (
                json.load(sys.stdin)
                if a.budget_file == "-"
                else json.loads(Path(a.budget_file).read_text())
            )
        tokens = a.tokens if a.tokens is not None else spec["tokens_actual"]
        data_dir = a.data_dir or spec.get("data_dir")
        if not data_dir:
            ap.error("准备数据必须指定 --data-dir，或在预算 JSON 中提供 data_dir")
        if not math.isfinite(tokens) or tokens <= 0:
            ap.error("tokens 必须是有限正数")
        meta = prepare(
            corpus_dir=a.corpus,
            data_dir=data_dir,
            tokens_needed=math.ceil(tokens * 1.02),
            dataset_name=a.dataset or spec.get("dataset_name", "fineweb"),
            seed_name=a.seed_name or spec.get("run_name", "corpus"),
            use_cache=not a.no_cache,
        )
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return

    sh = inventory(a.corpus, use_cache=not a.no_cache)
    tot = sum(s.tokens for s in sh)
    print(f"\n分片数   : {len(sh)}")
    print(
        f"总 tokens: {tot / 1e9:.2f}B (自报口径, 校正 {TOKEN_COUNT_SKEW} 后 {tot * TOKEN_COUNT_SKEW / 1e9:.2f}B)"
    )
    print(f"总行数   : {sum(s.rows for s in sh):,}")
    print(f"\n前 3 片: {', '.join(f'{s.name}={s.tokens / 1e9:.3f}B' for s in sh[:3])}")
    print(f"后 3 片: {', '.join(f'{s.name}={s.tokens / 1e9:.3f}B' for s in sh[-3:])}")


if __name__ == "__main__":
    main()
