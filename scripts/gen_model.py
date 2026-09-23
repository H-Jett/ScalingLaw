#!/usr/bin/env python3
"""按 Chinchilla 论文 Table A9 的规格生成 GPT-NeoX 模型目录。

只写 config.json + tokenizer 文件, **不写权重** —— 配合 LLaMA-Factory 的
`train_from_scratch: true` 从零初始化, 用于缩放律实验。

用法:
    python gen_model.py --out models/n355M --hidden 1024 --layers 20 --heads 16

规格来源: Hoffmann et al. 2022 (arXiv:2203.15556) Table A9。
表内规律: ffw = 4 x d_model, n_heads x head_dim = d_model。
"""

import argparse
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--hidden", type=int, required=True, help="d_model")
    ap.add_argument("--layers", type=int, required=True, help="n_layers")
    ap.add_argument("--heads", type=int, required=True, help="n_heads")
    ap.add_argument("--ffw", type=int, default=0, help="ffn 维度, 默认 4*hidden")
    ap.add_argument("--tokenizer-from", default="", help="取 tokenizer 文件的源目录")
    ap.add_argument("--vocab-size", type=int, default=50304)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--rotary-pct", type=float, default=0.25)
    a = ap.parse_args()

    ffw = a.ffw or 4 * a.hidden
    if a.hidden % a.heads != 0:
        raise SystemExit(f"hidden({a.hidden}) 不能被 heads({a.heads}) 整除, head_dim 必须是整数")
    head_dim = a.hidden // a.heads

    os.makedirs(a.out, exist_ok=True)
    cfg = {
        "architectures": ["GPTNeoXForCausalLM"],
        "model_type": "gpt_neox",
        "hidden_size": a.hidden,
        "num_hidden_layers": a.layers,
        "num_attention_heads": a.heads,
        "intermediate_size": ffw,
        "max_position_embeddings": a.seq_len,
        "vocab_size": a.vocab_size,
        "rotary_pct": a.rotary_pct,
        "rotary_emb_base": 10000,
        "layer_norm_eps": 1e-5,
        "hidden_act": "gelu",
        "use_parallel_residual": True,
        "attention_bias": True,
        "tie_word_embeddings": False,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "use_cache": True,
        "torch_dtype": "bfloat16",
    }
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for name in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(a.tokenizer_from, name)
        if not src or not os.path.exists(src):
            raise SystemExit(f"缺少 tokenizer 文件: {src}")
        shutil.copy(src, os.path.join(a.out, name))

    # HF GPTNeoX 实际参数量: 12*L*d^2 (attn+ffn) + 2*vocab*d (输入/输出 embedding, 不共享) + 13*L*d (LayerNorm)
    n_params = 12 * a.layers * a.hidden**2 + 2 * a.vocab_size * a.hidden + 13 * a.layers * a.hidden
    print(
        f"  {os.path.basename(a.out):>8}  d={a.hidden:<5} L={a.layers:<3} heads={a.heads:<3} "
        f"head_dim={head_dim:<4} ffw={ffw:<6} -> 参数 {n_params/1e6:.1f}M"
    )


if __name__ == "__main__":
    main()
