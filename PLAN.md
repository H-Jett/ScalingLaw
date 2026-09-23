# Chinchilla 缩放律复现计划

复现 Hoffmann et al. 2022《Training Compute-Optimal Large Language Models》(arXiv:2203.15556)
的**核心方法论**——用 isoFLOP 剖面测出 `N_opt ∝ C^a`、`D_opt ∝ C^b`，检验 `a ≈ b ≈ 0.5`。

模型架构用 **GPT-NeoX**（对齐 Pythia 系列），底座 `EleutherAI/pythia-14m` 仅取其 **tokenizer**，
尺寸通过 config 调整——每个尺寸一份 `config.json`，配 `train_from_scratch: true` 从零初始化。

## 0. 先明确这个计划能给什么、不能给什么

**能给：**
- 指数 `(a, b)` 的测量值，与论文的 (0.49, 0.51) 对比
- `L(N,D) = E + A/N^α + B/D^β` 的拟合（Approach 3）
- 一条属于自己算力预算的 compute-optimal 曲线

**不能给：**
- 论文的绝对 loss 数值 —— MassiveText **未公开**，不可复现。论文模型卡原文：
  "Primary Intended Users: DeepMind researchers. **We will not make this model available publicly.**"
- 论文原样的 20 tok/param 常数 —— tokenizer（vocab 32k BPE 无 NFKC vs Pythia 的 50304）改变了
  每 token 的信息量，即改变了有效 D。

## 1. 已实测的环境事实（2026-09-23）

| 项 | 值 | 来源 |
|---|---|---|
| GPU | 2 × H200 (sm90, 143GB) | nvidia-smi |
| torch | 2.8.0+cu128 | 实测 |
| 实测吞吐 | **1.07B 模型 / batch 32 → 44.6K tok/s ≈ 286 TFLOPS = 29% MFU** | 本机 50 步实测 |
| GPU 时价折算 | 20 单元 isoFLOP 单遍 ≈ **44 天**（单卡）；加超参候选 ≈ 70 天 | 算自 1.10e21 FLOPs |

⚠️ **注意力实现的坑**：`GPTNeoX` 在 transformers 5.8.0 里 **不支持 FlashAttention-2**
（`_supports_flash_attn_2` 未定义），只支持 SDPA / flex_attn。所以：
- 我们装的 flash-attn 2.8.3 **对本实验用不上**（对 LLaMA 架构仍有用）
- 29% MFU 就是 SDPA 下的实测值；若追求更高吞吐，需换架构或上 flex_attn

## 2. 尺寸阶梯（Chinchilla Table A9）

论文 Table A9 的规律：`ffw = 4 × d_model`，`n_heads × head_dim = d_model`。

⚠️ **A9 的 "Parameters (million)" 列与 HF 口径对不上**（A9 的 74M 行按 HF 算实为 113.6M）。
下表一律用 HF 口径（`12Ld² + 2·vocab·d + 13Ld`，vocab=50304），已用 LLaMA-Factory
实际报的 `1,067,137,792` 验证过一致。

| 名称 | d_model | L | heads | head_dim | ffw | 参数量 |
|---|---|---|---|---|---|---|
| n114M | 640 | 10 | 10 | 64 | 2560 | 113.6M |
| n205M | 768 | 18 | 12 | 64 | 3072 | 204.8M |
| n355M | 1024 | 20 | 16 | 64 | 4096 | 354.9M |
| n542M | 1280 | 21 | 10 | 128 | 5120 | 542.0M |
| n1067M | 1792 | 23 | 14 | 128 | 7168 | 1067.1M |
| n1470M | 2176 | 22 | 17 | 128 | 8704 | 1469.6M |
| n2303M | 2560 | 26 | 20 | 128 | 10240 | 2303.1M |

## 3. 训练网格

`N_opt = √(C/120)`（20 tok/param 锚点下 `C = 6ND` 且 `D = 20N` ⇒ `C = 120N²`）。
每个预算取其最近的 5 个尺寸，**必须跨过 N_opt 两侧**（否则抛物线拟合没有顶点）。

| 预算 C | N_opt | 尺寸阶梯（参数） | 该阶梯 tok/param 范围 |
|---|---|---|---|
| 1.0e19 | 289M | 114M / 205M / 355M / 542M / 1067M | 129 → 1.5 |
| 3.0e19 | 500M | 205M / 355M / 542M / 1067M / 1470M | 119 → 2.3 |
| 6.0e19 | 707M | 355M / 542M / 1067M / 1470M / 2303M | 79 → 1.9 |
| 1.2e20 | 1000M | 355M / 542M / 1067M / 1470M / 2303M | 159 → 3.8 |

共 **20 个实验单元**，单遍开销 **1.10e21 FLOPs**（= 4 个预算之和 × 各自 5 个尺寸）。

> 若后续要加第 5 个预算 3e20（N_opt=1.6B），因为 `D` 超 100BT（需 2.5e11 token）必须先用
> `sample/350BT` 扩语料，且单遍开销会涨到 ~2.6e21。

## 4. 数据

**`HuggingFaceFW/fineweb-edu` 的 `sample/100BT` 子集**（HF 上三个子集都在：`10BT`/`100BT`/`350BT`）。

为什么不需要凑 math/code 配比：论文 **App. C / Table A2** 专门做了跨数据集对照——单用 C4 得
`(a,b)=(0.50,0.50)`，单用 GitHub code 得 `(0.53,0.47)`，与主结论一致。论文结论：
**只要训练不超过 1 个 epoch，缩放律与数据集无关。**

100BT 是**单次 run** 的上限（超过 1 epoch 就要引入 datablock 的重复数据修正），
多个 run 可以复用同一份语料。上表最大 token 需求 56.35B，留有充足余量。

token 数按 `D = C/6N` 反推，cosine 周期长度**必须精确等于 max_steps**（论文 App. B：
超出 25% 性能明显下降）。`make_grid.py` 据此生成 `max_steps`。

## 5. 实施步骤

### 阶段 0：语料准备（~0.5 天）
```bash
export HF_HOME=/volume/data/hjiang02/.cache/huggingface
# 下 fineweb-edu sample/100BT，转成 LLaMA-Factory 的 pt 格式（text 字段），落 /volume/data
```
产出 `data/chinchilla_corpus/` + `dataset_info.json` 条目 `chinchilla_corpus`（columns: prompt←text）。

### 阶段 1：小预算先验证闭环（1 个单元，~1 天）
只跑 `C=1e19 × n355M`（最接近 N_opt 的点，D=4.7e9 tokens）。目的：
- 确认 `train_from_scratch` 路径、tokenizer/packing、loss 曲线形状正常
- 校准实际吞吐与显存，修正阶段 2 的 batch 设定
**不要跳过**——这是唯一能低成本发现配置错误的机会。

### 阶段 2：**按尺寸调 batch / LR / β2**（~1 周）⚠️ 不可跳过

**这是整个计划最容易失败的地方。** 论文没有公布每个尺寸的 LR/batch/β2
（grep 过全文，App. D.1 只给了 max LR 区间 2e-4 小 → 1.25e-4 大，App. D.4 给了对照实验的
0.5M token batch / 1.5e-4 LR）。Porian et al. 2024 (arXiv:2406.19146) 的实测结论：
**小尺度上若不按尺寸调这三项，拟合出的 `a` 会系统性偏大到 0.6~0.9，复现不出 0.5。**

要调的（对每个尺寸，用 1e19 预算下的 D 短跑几个点）：
- **batch size（按 token 计）**：随模型增大而增大
- **max LR**：经验上 `LR ∝ 1/width`；起点用论文区间，在各自尺寸上扫 3 个值
- **AdamW β2**：Porian 指出 β2=0.95 在小 batch(≤128) 下不是最优，**要用 0.99/0.999**
- **grad clip 1.0**（论文 Gopher §3.2；7.1B 以上降到 0.25）
- warmup：论文用固定 1500 步；Porian 建议 warmup tokens ≈ N，小模型上按后者更稳

### 阶段 3：跑满 20 个单元（随预算）
`make_grid.py` 生成配置后逐个跑。**每跑完一个尺寸就立刻拟合抛物线**，不要等全部跑完
——如果发现某条曲线没有顶点（阶梯没跨过最优点），要能及时补点而不是返工。

### 阶段 4：拟合与对照（~2 天）
- 每条 isoFLOP 曲线拟合抛物线 → 顶点即 `N_opt(C)`、`D_opt(C)`
- 对 `(C, N_opt)` 和 `(C, D_opt)` 拟合幂律 → 得到 `(a, b)`
- 用全部终点 loss 拟合 `L(N,D) = E + A/N^α + B/D^β`（Approach 3）
- 对照：论文 Approach 2 的 (0.49, 0.51)；Epoch AI 复现的修正指数 (α≈β≈0.35)

## 6. 怎么判断"做成了"

| 检查项 | 期望 | 说明 |
|---|---|---|
| 每条 isoFLOP 曲线有明确 U 形谷 | 必须 | 没有谷 = 阶梯没跨过最优点，补点重跑 |
| 拟合出的 `a`、`b` | 落在 0.4~0.6 | 论文 0.49/0.51；偏差大先怀疑阶段 2 没调好 |
| `a + b ≈ 1` | 应成立 | 由 `C = 6ND` 约束推出的守恒关系 |
| `a ≈ b` | 应成立 | 即 20 tok/param 的由来 |

**若 `a` 落在 0.6~0.9**：几乎一定是阶段 2 的按尺寸调参没做够（Porian 的结论），不是实现 bug。

## 7. 目录结构

```
Chinchilla-Scaling-Law/
  PLAN.md                     # 本文件
  scripts/
    gen_model.py              # 按 Table A9 规格生成 GPT-NeoX 模型目录(config+tokenizer, 无权重)
    make_grid.py              # 生成模型目录 + 训练 YAML + 清单 reports/grid.json
    run_grid.py               # (阶段 3 再写) 批量执行 + 断点续跑
    fit_scaling.py            # (阶段 4 再写) 抛物线拟合 + 幂律拟合
  configs/                    # 由 make_grid.py 生成的训练配置
  models/                     # 由 make_grid.py 生成的各尺寸模型目录
  data/                       # 语料(符号链接到 /volume/data 下的实际存放处)
  logs/                       # 训练日志
  reports/                    # grid.json、拟合结果(带时间戳)
```

## 8. 复用本机已有的 LLaMA-Factory 环境

```bash
export HF_HOME=/volume/data/hjiang02/.cache/huggingface   # 登录 shell 里没设, 需手动
export DISABLE_VERSION_CHECK=1
LLAMA_FACTORY=/volume/data/hjiang02/workspace/ScalingLaw/submodules/LLaMA-Factory
llamafactory-cli train <config.yaml>
```

环境已装好并验证：torch 2.8.0+cu128 / vllm 0.11.0 / flash-attn 2.8.3 / deepspeed 0.18.4，
`train_from_scratch` 路径已实测跑通（从 config 随机初始化，loss 从 ~10.4 正常下降）。

## 9. 关键参考

- 原文：arXiv:2203.15556（Table A9 尺寸表、App. B cosine、App. C 跨数据集、App. F FLOPs 计数）
- Epoch AI 复现：arXiv:2404.10102 —— 重拟合得 α≈β≈0.35（修正指数与 20 tok/param 自洽）
- Porian et al. 2024：arXiv:2406.19146 —— 调参对指数测量的影响（阶段 2 的依据）
- datablations：arXiv:2305.16264 —— 重复数据下的缩放律修正（若超 1 epoch 必读）
- FloydHub/nanoGPT 相关实现：`epoch-research/analyzing-chinchilla` 有从论文图里像素级反推的 245 个数据点
