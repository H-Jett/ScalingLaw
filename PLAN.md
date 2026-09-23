# Chinchilla 缩放律复现计划

复现 Hoffmann et al. 2022《Training Compute-Optimal Large Language Models》(arXiv:2203.15556)
的**核心方法论**——用 isoFLOP 剖面测出 `N_opt ∝ C^a`、`D_opt ∝ C^b`，检验 `a ≈ b ≈ 0.5`。

模型架构用 **GPT-NeoX**(对齐 Pythia 系列)，**以 `EleutherAI/pythia-14m` 为模板**：只取其
config + tokenizer，尺寸通过改 4 个字段往上派生，配 `train_from_scratch: true` 从零初始化。
选 Pythia 作模板不是为了"复现 Pythia"，而是因为它是这批模型里**唯一有官方公开权重和完整
训练曲线**的家族，可以拿来做独立交叉验证。

## 0. 先明确这个计划能给什么、不能给什么

**能给：**
- 指数 `(a, b)` 的测量值，与论文的 (0.49, 0.51) 对比
- `L(N,D) = E + A/N^α + B/D^β` 的拟合（Approach 3）
- 一条属于自己算力预算的 compute-optimal 曲线

**不能给：**
- 论文的绝对 loss 数值 —— MassiveText **未公开**，不可复现。论文模型卡原文：
  "Primary Intended Users: DeepMind researchers. **We will not make this model available publicly.**"
- 论文原样的 20 tok/param 常数 —— 换了 tokenizer(vocab 50304 = Pythia 口径)就改变了每 token
  的信息量，即改变了有效 `D`

**关键推论**：既然绝对数值注定对不上，架构上追求"忠实于 Gopher/Chinchilla"就买不到可比性。
且指数对架构是弱敏感的（论文 App. C 演示了跨数据集指数几乎不动）。**所以架构选型是工程
效率问题，不是保真度问题。**

## 1. 环境事实（2026-09-23 实测）

| 项 | 值 | 来源 |
|---|---|---|
| GPU | **4 × H100 80GB** (HBM3) | `nvidia-smi` |
| 系统 python | `/usr/bin/python3` = 3.12 + torch 2.6.0+cu124 + transformers 4.51.3 | 实测（**不要动，另建环境**） |
| conda | `/opt/miniconda3`（新建）+ 环境 `scalinglaw`，见 §8 | 见 §8 |
| ⚠️ `/volume/data` | **配额已满，仅剩 ~776M**，写 1GB 在 700M 处报 `Disk quota exceeded` | 实测 |
| `/` (overlay) | ~247G 可用，**但容器重建即丢** | `df -h` |

⚠️ **注意力实现**：`GPTNeoX` 在 **transformers 4.51.3 里 `_supports_flash_attn_2 = True`**
（旧版本 5.8.0 下不声明支持，那是版本特性，不是架构限制）。是否真的能跑 FA2 需在
`scalinglaw` 环境里实测；`_supports_flex_attn` 也在。**当前 MFU 未知**，需复测。

## 2. 模型尺寸阶梯

由 `scripts/make_model_ladder.py` 从 pythia-14m 模板派生，**第 1 档就是模板本身**（原样保留，
它是长宽比律的锚点，也是唯一能对官方曲线做交叉验证的尺寸）。

| 目录 | d_model | L | heads | head_dim | ffw | N_非emb | N_总 | emb占比 |
|---|---|---|---|---|---|---|---|---|
| `pythia-14m` | 128 | 6 | 4 | 32 | 512 | 1.2M | 14.1M | 91.5% |
| `pythia-23m` | 192 | 8 | 6 | 32 | 768 | 3.6M | 22.9M | 84.4% |
| `pythia-43m` | 320 | 9 | 10 | 32 | 1280 | 11.1M | 43.3M | 74.4% |
| `pythia-86m` | 512 | 11 | 16 | 32 | 2048 | 34.7M | 86.2M | 59.8% |
| `pythia-184m` | 768 | 15 | 12 | 64 | 3072 | 106.3M | 183.6M | 42.1% |
| `pythia-442m` | 1216 | 18 | 19 | 64 | 4864 | 319.7M | 442.0M | 27.7% |
| `pythia-1.2b` | 1920 | 23 | 15 | 128 | 7680 | 1.02B | 1.21B | 15.9% |

**完整尺寸表 + 设计规则见 `models/configs/README.md`（脚本自动生成）。** 要点：

1. **骨架公式**（已对 pythia 官方 9 档校验，误差 0）：
   `N_非emb = 12·L·d² + 13·L·d + 2·d`，`N_总 = N_非emb + 2·vocab·d`
2. **长宽比律**：`L ∝ d^0.5`（等价 `d ∝ N^0.4, L ∝ N^0.2`）。放大模型主要靠放大 `d`
   （平方项），但 `d` 和 `L` 必须一起长
3. **离散约束**：`d` 为 64 的倍数、`head_dim = clamp(≥d/16 的最小 2 的幂, 32, 128)`、
   `ffw = 4·d`。量化后**不可能同时精确命中参数量和理想长宽比**，脚本用
   `|log(N/N_target)| + λ·|log(L/L_ideal)|` 折中，表里两个偏差都列出来
4. ⚠️ **口径**：目录名用**总参数量**（pythia/HF 命名口径），阶梯按**非 embedding**参数量定。
   这个区分必须记住 —— `pythia-14m` 名义 14M，但其中 12.9M 是词表，计算层只有 1.2M

## 3. 实验网格（待定稿）

isoFLOP 剖面法的要求：每个预算 `C` 下扫一串尺寸，**必须有一个尺寸落在 `N_opt` 两侧**
（否则抛物线没有顶点，测不出最优点）。

上一版网格（Chinchilla Table A9 尺寸梯）已废弃，因为 A9 的尺寸与我们这条 pythia 阶梯无关。
**新版网格待定**，需要：
- 选定 3~4 个 isoFLOP 预算 `C`
- 每个 `C` 从 §2 的 7 档里挑 4~5 档，保证跨过 `N_opt = √(C/120)` 两侧
- token 数由 `D = C/(6N)` 反推，**cosine 周期长度必须精确等于 `max_steps`**
  （论文 App. B：超出 25% 性能明显下降）

⚠️ 注意 `N` 的口径与 §2 一致（建议用 `N_总`，因为 `C = 6ND` 的 `C` 是含 embedding 的算力；
论文原文："We include all training FLOPs, including those contributed to by the embedding
matrices" 以及 "we also count embeddings matrices in the total parameter count"）。
但我们 vocab 是 50304（比论文 32k 大 1.57 倍），小模型端 embedding 占比会到 74~92%，
论文"embedding 贡献很小"的前提在小端**不成立** —— 这里要做敏感性分析：两个口径各算一遍，
看谷底位置移动多少。

## 4. 数据

**`HuggingFaceFW/fineweb-edu` 的 `sample/100BT` 子集**（HF 上三个子集都在：`10BT`/`100BT`/`350BT`）。

为什么不需要凑 math/code 配比：论文 **App. C / Table A2** 专门做了跨数据集对照——单用 C4 得
`(a,b)=(0.50,0.50)`，单用 GitHub code 得 `(0.53,0.47)`，与主结论一致。论文结论：
**只要训练不超过 1 个 epoch，缩放律与数据集无关。**

100BT 是**单次 run** 的上限（超过 1 epoch 就要引入 datablock 的重复数据修正），
多个 run 可以复用同一份语料。

⚠️ **磁盘**：tokenized 语料的体积是几百 GB 量级，当前 `/volume/data` 配额已满，**阶段 0 之前
必须先解决**（申请配额 / 清理 / 换存储）。

## 5. 实施步骤

### 阶段 0：环境 + 语料（~1 天）
- miniconda + `scalinglaw` 环境（见 §8）
- 下 fineweb-edu `sample/100BT`，转成 LLaMA-Factory 的 pt 格式（text 字段）
- 产出 `data/chinchilla_corpus/` + `dataset_info.json` 条目

### 阶段 1：小预算先验证闭环（1 个单元，~1 天）
只跑一个小单元（比如 `pythia-86m` 或 `pythia-184m`）。目的：
- 确认 `train_from_scratch` 路径、tokenizer/packing、loss 曲线形状正常
- **复测吞吐与 MFU**（旧记录 286 TFLOPS/29% 是在 2×H200 + transformers 5.8 上测的，已失效），
  据此修正阶段 2 的 batch 设定
- 顺便用 `pythia-14m` 这一档对齐官方曲线，验证实现正确性
**不要跳过。**

### 阶段 2：**按尺寸调 batch / LR / β2**（~1 周）⚠️ 不可跳过

**这是整个计划最容易失败的地方。** 论文没有公布每个尺寸的 LR/batch/β2（App. D.1 只给了
max LR 区间 2e-4 小 → 1.25e-4 大）。Porian et al. 2024 (arXiv:2406.19146) 的实测结论：
**小尺度上若不按尺寸调这三项，拟合出的 `a` 会系统性偏大到 0.6~0.9，复现不出 0.5。**

要调的（对每个尺寸，用最短的预算短跑几个点）：
- **batch size（按 token 计）**：随模型增大而增大
- **max LR**：经验上 `LR ∝ 1/width`；起点用论文区间，在各自尺寸上扫 3 个值
- **AdamW β2**：Porian 指出 β2=0.95 在小 batch(≤128) 下不是最优，**要用 0.99/0.999**
- **grad clip 1.0**（论文 Gopher §3.2；7.1B 以上降到 0.25）
- **warmup**：论文用固定 1500 步；Porian 建议 warmup tokens ≈ N，小模型上按后者更稳
- ⚠️ **初始化**：HF 的 `GPTNeoXPreTrainedModel._init_weights` 对**所有尺寸**都用固定
  `initializer_range=0.02`，**没有** EleutherAI 原实现里 `1/√(2L)` 的残差输出缩放。
  含义：① 大 `L` 档若 loss 一开头就炸，先怀疑 init 而不是自己写错了；② 在 HF 里
  from-scratch 训出来的模型与 pythia 官方权重**不是同一套 init 方案**，交叉验证要留这个偏差

### 阶段 3：跑满网格（随预算）
生成配置后逐个跑。**每跑完一个尺寸就立刻拟合抛物线**，不要等全部跑完——如果发现某条曲线
没有顶点（阶梯没跨过最优点），要能及时补点而不是返工。

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
ScalingLaw/
  PLAN.md                       # 本文件
  scripts/
    make_model_ladder.py        # 从 pythia-14m 模板派生尺寸阶梯(config+tokenizer, 无权重)
    # run_grid.py               # (阶段 3 再写) 批量执行 + 断点续跑
    # fit_scaling.py            # (阶段 4 再写) 抛物线拟合 + 幂律拟合
  models/
    templates/pythia-14m/       # 下载的官方模板(尺寸设计的锚点)
    configs/                    # 生成的各尺寸 config(tokenizer 为软链接)
      README.md                 # 尺寸表 + 设计规则(脚本自动生成)
  configs/                      # (阶段 3) 训练 YAML
  data/                         # 语料(符号链接到实际存放处)
  logs/                         # 运行日志
  reports/                      # 拟合结果(带时间戳)
  submodules/LLaMA-Factory/     # 训练框架(submodule, 需 init)
```

⚠️ **软链接注意**：`models/configs/*/tokenizer.json` 是指向 `models/templates/` 的软链接。
新 clone 后软链接会悬空，**必须先跑一次 `make_model_ladder.py` 把模板下下来**。

## 8. 环境与运行

```bash
# conda 装在 /opt（overlay），不在 /volume/data
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate scalinglaw

export DISABLE_VERSION_CHECK=1
LLAMA_FACTORY=/volume/data/hjiang02/workspace/ScalingLaw/submodules/LLaMA-Factory
llamafactory-cli train <config.yaml>
```

⚠️ **持久性警告**：`/opt` 在容器 overlay 层，**容器被驱逐/重建会连同环境一起丢失**。
理想位置是 `/volume/data/hjiang02/`（持久），但该盘配额已满放不下。**拿到配额后应把
miniconda 迁到 `/volume/data` 下重建。**

## 9. 关键参考

- 原文：arXiv:2203.15556（Table A9 尺寸表、App. B cosine、App. C 跨数据集、App. F FLOPs 计数）
- Epoch AI 复现：arXiv:2404.10102 —— 重拟合得 α≈β≈0.35（修正指数与 20 tok/param 自洽）
- Porian et al. 2024：arXiv:2406.19146 —— 调参对指数测量的影响（阶段 2 的依据）
- datablations：arXiv:2305.16264 —— 重复数据下的缩放律修正（若超 1 epoch 必读）
- `epoch-research/analyzing-chinchilla`：从论文图里像素级反推的 245 个数据点
- Pythia：arXiv:2304.01373 —— 模板来源，也是交叉验证的基准曲线
