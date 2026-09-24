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

## 1. 环境事实（2026-09-24 实测，已验证可训练）

| 项 | 值 | 来源 |
|---|---|---|
| GPU | **4 × H100 80GB HBM3**，卡间 **NVLink**（`nvidia-smi topo -m` 全 `NV18` = 18 条链路） | `nvidia-smi` |
| CPU / 内存 | **192 核 / 2.0 TB** | `nproc` / `free -g` |
| `/dev/shm` | **880 GB**（多卡 dataloader 不会撞 shm 上限） | `df -h` |
| 驱动 / CUDA | 570.133.20 / CUDA 12.9 | `nvidia-smi` |
| Python | `/usr/bin/python3` = **3.12.3**，**就是这套**（无独立 conda 环境） | 实测 |
| torch | **2.8.0+cu128**（`cxx11abiFALSE`） | 实测 |
| transformers | **5.8.0** | 实测 |
| flash-attn | **2.8.3**（已装，`import` 可用） | 实测 |
| 其它 | deepspeed 0.18.4 / vllm 0.11.0 / peft 0.18.1 / accelerate 1.11.0 / datasets 4.0.0 / torchdata 0.11.0 / bitsandbytes 0.50.2 / triton 3.4.0 | `pip list` |
| LLaMA-Factory | **0.9.5**，见 §8（submodule + 符号链接） | 实测跑通 |
| ⚠️ `/volume/data` | **配额已满，仅剩 ~764M**，写 1GB 在 700M 处报 `Disk quota exceeded` | 实测 |
| `/` (overlay) | ~183G 可用，**但容器重建即丢** | `df -h` |
| 容器 | pod `dev-scalinglaw-151a86d5-master-0`，`PET_NNODES=1`（**开发容器，非训练任务容器**） | `env` |

⚠️ **这套环境是平台配额配好的，各依赖全卡在上界，是自洽的一套 —— 不要重装、不要升级**：

```
transformers                     5.8.0     (LLaMA-Factory 0.9.6.dev0 允许 <=5.8.0)
accelerate                      1.11.0     (peft 要求 <=1.11.0)
datasets                         4.0.0     (要求 <=4.0.0)
peft                            0.18.1     (要求 <=0.18.1)
torchdata                       0.11.0     (要求 <=0.11.0)
```

**注意 `transformers 5.8.0` 与我们的 submodule（LLaMA-Factory **0.9.5**，
声明 `<=5.6.0`）存在版本错位** —— 但 **实测能正常工作**（见下）。

### 1.1 实测结论：能训练

**端到端训练已跑通**（用 `models/configs/pythia-14m` 的 config + `train_from_scratch`）：

```
Number of trainable parameters = 14,067,712        ← 与官方 pythia-14m 一致
loss 10.77 → 9.627；模型正常保存
4 卡: "Initializing 4 distributed tasks"，exit 0，22.83 steps/s（与单卡一致）
```

**吞吐实测**（`pythia-184m`，d=768 / L=15，bf16，训练步）：

| 配置 | tok/s | MFU | 相对 SDPA |
|---|---|---|---|
| seq=2048, `sdpa` | 218,564 | 24.3% | 1.00× |
| seq=2048, `flash_attention_2` | 221,959 | 24.7% | 1.02× |
| seq=2048, `flex_attention` | 221,395 | 24.7% | 1.01× |

**FA2 的收益已扫过序列长度：速度上几乎没有，真正的价值是显存。**
（`pythia-184m`，bf16，真实训练步；每档 batch 取到不 OOM 为止）

| seq | batch | SDPA (tok/s / MFU) | FA2 (tok/s / MFU) | FA2 加速比 |
|---|---|---|---|---|
| 2048 | 16 | 227,793 / 25.4% | 231,080 / 25.7% | 1.01× |
| 4096 | 8 | 200,847 / 22.4% | 204,761 / 22.8% | 1.02× |
| 8192 | 4 | 162,767 / 18.1% | 166,736 / 18.6% | 1.02× |
| **8192** | **8** | **OOM** | **170,771 / 19.0%** | — |
| 16384 | 2 | 119,622 / 13.3% | 123,370 / 13.7% | 1.03× |

两个结论：

1. **FA2 只快 1~3%，连 seq=16384 也一样** —— 所以"长序列才体现 FA2 优势"这个常见说法
   **在本配置下不成立**。别指望 FA2 提速。
2. **FA2 的真正价值是显存**：seq=8192 时 SDPA 在 batch=8 直接 OOM，FA2 能跑。
   即 FA2 换来的是**更大的 batch**，不是更快的算子。
3. 附带发现：**MFU 随序列长度单调下降**（25.4% → 13.3%），token 吞吐也一样降。
   所以 **seq=2048 是这套配置下效率最高的选择** —— 正好与 Chinchilla 论文用的 2048 一致。

**架构层事实**（用**正确**的属性名 `_supports_flash_attn` 查，实测 `gpt_neox` 与 `llama`
都是 `True`）：

| 版本 | `_supports_flash_attn_2` | `_supports_flash_attn` | `_supports_flex_attn` | `_supports_sdpa` |
|---|---|---|---|---|
| 4.51.3 | `True` | — | — | `True` |
| 4.57.1 | 不存在 | **`True`** | `True` | `True` |
| 5.8.0（本环境） | 不存在 | **`True`** | `True` | `True` |
| 5.17.0 | 不存在 | **`True`** | `True` | `True` |

⚠️ **别用 `_supports_flash_attn_2` 判断 FA2 支持** —— 该属性在 5.x 里**改名**为
`_supports_flash_attn`。本项目此前两度因此得出错误结论（旧版说"GPTNeoX 不支持 FA2"，
是误判）。**GPTNeoX 支持 FA2**，实测可用。

**MFU 天花板 ~25~27%**（对 GPT-NeoX 小模型），已在两套不同环境里**独立复现两次**，
所以不是环境问题，是架构 + 小尺寸的固有特性。**注意力实现不是主要杠杆**，真正该调的是
batch / seq / 宽度。

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
- **环境已就绪，无需搭建**（见 §8：用平台自带 `/usr/bin/python3`，不要建 conda）
- **先解决 `/volume/data` 配额**（见 §4 ⚠️）—— 这是**唯一的硬阻塞**
- 下 fineweb-edu `sample/100BT`，转成 LLaMA-Factory 的 pt 格式（text 字段）
- 产出 `data/chinchilla_corpus/` + `dataset_info.json` 条目

### 阶段 1：小预算先验证闭环（1 个单元，~1 天）
只跑一个小单元（比如 `pythia-86m` 或 `pythia-184m`）。目的：
- 确认 `train_from_scratch` 路径、tokenizer/packing、loss 曲线形状正常
- **扫 batch / seq 把 MFU 提上去**。见 §1 已测数据：MFU 天花板 ~25~27%，且
  **seq=2048 时最高（25.4%）并随序列长度单调下降**，所以起点就用 **seq=2048**；
  再往上调的是**宽度 / batch**，不是注意力实现（三种注意力实现差异只有 1~2%）
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
  submodules/LLaMA-Factory/     # 训练框架(submodule, 需 init, 见 §8.1)
```

⚠️ **软链接注意**：`models/configs/*/tokenizer.json` 是指向 `models/templates/` 的软链接。
新 clone 后软链接会悬空，**必须先跑一次 `make_model_ladder.py` 把模板下下来**。
另外 `workspace/LLaMA-Factory -> submodules/LLaMA-Factory` 这条链接（§8.1）也需要重建。

## 8. 环境与运行

**结论（2026-09-24）：不建独立环境，直接用平台自带的 `/usr/bin/python3`。**
下面这套是平台镜像配额配好的，**自洽且已验证可训** —— 不需要 conda，也**不要重装/升级任何包**。

```bash
# 跑训练（无需 activate，无 conda）
export DISABLE_VERSION_CHECK=1          # 版本错位的豁免（见下）
cd /volume/data/hjiang02/workspace/ScalingLaw
llamafactory-cli train <config.yaml>     # 或 python -m llamafactory.cli train <config.yaml>
```

| 项 | 值 |
|---|---|
| Python | `/usr/bin/python3` = **3.12.3** |
| torch | **2.8.0+cu128**（`cxx11abiFALSE`；CUDA 12.8，`torch.cuda.device_count() == 4`） |
| transformers | **5.8.0** |
| flash-attn | **2.8.3**（`import flash_attn` 可用） |
| datasets / accelerate / peft / deepspeed | 4.0.0 / 1.11.0 / 0.18.1 / 0.18.4 |
| triton / tokenizers / safetensors / sentencepiece / einops | 3.4.0 / 0.22.2 / 0.5.3 / 0.2.2 / 0.8.1 |
| 推理/其它 | vllm 0.11.0 / bitsandbytes 0.50.2 |

⚠️ 一个小瑕疵：`import vllm` 会警告 `torchao 0.15.0` 与 `torch 2.8.0` 不兼容（缺 torchao 的
编译扩展）。**只影响 torchao 那部分量化路径，vllm 本身能 import** —— 阶段 4 若要批量算
loss 用它，先在小样本上验一遍。

### 8.1 LLaMA-Factory 的接线方式

框架是 submodule（`submodules/LLaMA-Factory`，**0.9.5**），用**符号链接**接回来：

```
/volume/data/hjiang02/workspace/LLaMA-Factory -> .../ScalingLaw/submodules/LLaMA-Factory
```

之所以需要这条链接：早期 pip 以 editable 方式装过 LLaMA-Factory，其 `.pth` 指向
`.../workspace/LLaMA-Factory/src`。submodule 没 init 时该路径不存在，`import llamafactory`
直接失败。**新 clone 后必须 `git submodule update --init --recursive` 再做这条符号链接**，
否则 CLI 起不来。

### 8.2 版本错位是已知且无害的

| 包 | 装了的 | LLaMA-Factory 0.9.5 声明 | 实测 |
|---|---|---|---|
| transformers | **5.8.0** | `<=5.6.0` | ✅ 正常（0.9.6.dev0 的上界正好是 5.8.0） |
| datasets | 4.0.0 | `<=4.0.0` | ✅ |
| accelerate | 1.11.0 | peft 要求 `<=1.11.0` | ✅ |
| peft | 0.18.1 | `<=0.18.1` | ✅ |
| torchdata | 0.11.0 | `<=0.11.0` | ✅ |

各依赖**全部卡在自己允许区间的上界** —— 这是平台有意锁的版本矩阵，动一个就会连锁崩。
transformers 5.8.0 超出 0.9.5 声明的 `<=5.6.0`，但**实测功能正常**（0.9.6.dev0 才把上界
提到 5.8.0，说明 5.8.0 本身没有破坏性改动）。**因此不要按 0.9.5 的声明去降级
transformers** —— 那会把 torch/accelerate 一起拖下水。需要时用 `DISABLE_VERSION_CHECK=1`
跳过运行时版本告警。

⚠️ **不要 `pip install -e .` LLaMA-Factory**，同理**不要 `pip install -r` 它的 requirements**：
两者都会按 pyproject 重新解析依赖，把上面这个矩阵改掉。

### 8.3 曾经的失败路线（别再走一遍）

1. **容器重建会抹掉 overlay**：这个项目已经因此丢过一次整个 `/opt/miniconda3`（torch 2.6.0
   + flash-attn 2.7.4 那套自建环境）。**别把环境建在 `/opt`、`/root`、`/tmp` 这些
   overlay 路径上**。平台镜像里的 `/usr/bin/python3` 不在这个风险里
2. **自己装 flash-attn 是多余的**：环境已带 2.8.3。若真要从零装（比如换个 torch 版本），
   走预编译 wheel，**别去 build 源码**；`pip install` 时务必 `--no-deps`，否则 pip 会去
   "修" torch / transformers。wheel 文件名四个字段要和环境逐项对齐：
   `cp312` + `torch2.8` + `cu12` + `cxx11abiFALSE`
   （`python -c "import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)"`）
3. **conda 26.x 的 ToS**：非交互 `conda create` 会要求 accept `repo.anaconda.com` 服务条款，
   而这是法律层面由你自己做的同意，我不代 accept

**已实测训练闭环**：用 `models/configs/pythia-14m` 的 config 建模型（参数
**14,067,712**，与官方 pythia-14m 完全一致），在 GPU 上 loss 从 `ln(50304) = 10.83`
附近（10.77）起步并正常下降 —— 说明 tokenizer 与初始化都正确。4 卡
（`FORCE_TORCHRUN=1`）实跑 exit 0，NCCL 总线带宽 281.5 GB/s。

## 9. 关键参考

- 原文：arXiv:2203.15556（Table A9 尺寸表、App. B cosine、App. C 跨数据集、App. F FLOPs 计数）
- Epoch AI 复现：arXiv:2404.10102 —— 重拟合得 α≈β≈0.35（修正指数与 20 tok/param 自洽）
- Porian et al. 2024：arXiv:2406.19146 —— 调参对指数测量的影响（阶段 2 的依据）
- datablations：arXiv:2305.16264 —— 重复数据下的缩放律修正（若超 1 epoch 必读）
- `epoch-research/analyzing-chinchilla`：从论文图里像素级反推的 245 个数据点
- Pythia：arXiv:2304.01373 —— 模板来源，也是交叉验证的基准曲线
