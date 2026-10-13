# Chinchilla 缩放律复现计划

## 这份计划要做什么：**先测，再看结论**

我们**在动手之前不去读 Chinchilla 的最终答案**。所有超参、尺寸、token 预算都由自己
的实验设计推出来；论文的 (0.49, 0.51)、20 tok/param、`√(C/120)` 这些东西**只允许在
最后一步拿来对照**，不允许用来决定实验怎么跑。

理由很实在：如果先知道"答案应该是 0.5，而且谷底在 20 tok/param"，那么每一步都会不自觉
地朝那个方向调——

- 谷底没出来，就会想"再补个点应该就有了"，而不是记录下来"这里没有谷底"
- 拟合出 0.42，就会想"是不是 LR 没调好"，而不是问"0.42 是不是真实结果"
- 试到第 3 个 token 预算时，已经知道该往哪边挪，等于把答案喂给了过程

这种做法测出来的**不是缩放律，是"我能不能调出论文那个数"**。而缩放律的价值恰恰在于它是
一条**先验未知、可被证伪**的经验规律；先看答案会让实验失去证伪能力。

### 具体怎么保证

| 规矩 | 为什么 |
|---|---|
| **超参独立定**：LR / batch / β2 / warmup 全按自己测出来的最优值定（阶段 2 的扫描），不抄论文 | 抄了就把论文的结论引进来了 |
| **预算与尺寸网格先写死再跑**：§3 的网格一旦定稿就冻结，跑完再改 | 防止"看到中间结果就调网格"这种事后选择 |
| **允许也鼓励出现非 0.5 的结果**：`a` 落在 0.6~0.9 是**有效发现**，不是 bug | Porian et al. 的实测就是调参不足会系统性偏大 |
| **每一档预算都留存原始 loss**：`runs/*/summary.json` 与 `trainer_log.jsonl` 全留 | 事后想换口径/换拟合方式时，不用重跑 |
| **论文结论只在 §6 做对照**，不参与任何决策 | 对照是终点，不是起点 |

⚠️ 一个必须诚实面对的后果：**这样做出来的指数很可能不是 0.5**。算力跨度只有 16×（§3），
`vocab=50304` 又让 embedding 在小模型端占到 91.5%（§3 的口径敏感性）。若最后测出
`a ≈ 0.4` 或 `a ≈ 0.7`，**正确做法是把测量值连同不确定性一起报出来、并分析为什么偏移**，
而不是回头把网格调到"能出 0.5 为止"。这份计划的目标是**做一次诚实的测量**。

---

## 0. 这个计划能给什么、不能给什么

**能给：**
- 指数 `(a, b)` 的**测量值**（含置信区间），**以及它是否等于 0.5 这个检验结果**
- `L(N,D) = E + A/N^α + B/D^β` 的拟合
- 一条属于自己算力预算的 compute-optimal 曲线
- 若与论文不符：**差异归因**（是调参不足？是 embedding 口径？是算力跨度太窄？）

**不能给：**
- 论文的绝对 loss 数值 —— MassiveText **未公开**，不可复现。论文模型卡原文：
  "Primary Intended Users: DeepMind researchers. **We will not make this model available publicly.**"
- 论文原样的 20 tok/param 常数 —— 换了 tokenizer(vocab 50304 = Pythia 口径)就改变了每 token
  的信息量，即改变了有效 `D`

**关键推论**：既然绝对数值注定对不上，架构上追求"忠实于 Gopher/Chinchilla"就买不到可比性。
且指数对架构是弱敏感的（跨数据集对照实验显示指数几乎不动）。**所以架构选型是工程效率
问题，不是保真度问题。**

模型架构用 **GPT-NeoX**(对齐 Pythia 系列)，**以 `EleutherAI/pythia-14m` 为模板**：只取其
config + tokenizer，尺寸通过改 4 个字段往上派生，配 `train_from_scratch: true` 从零初始化。
选 Pythia 作模板不是为了"复现 Pythia"，而是因为它是这批模型里**唯一有官方公开权重和完整
训练曲线**的家族，可以拿来做交叉验证（**注意：是验证实现有没有写错，不是校准指数**）。

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

由 `scripts/generate_model_configs.py` 从 pythia-14m 模板派生，**第 1 档就是模板本身**（原样保留，
它是长宽比律的锚点，也是唯一能对官方曲线做交叉验证的尺寸）。`vocab_size`、长宽比锚点
`(d0, L0)` 都从模板 config 里读，所以换模板就能换一整套阶梯。

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

## 3. 实验网格（⚠️ **必须定稿并冻结后再开跑**）

isoFLOP 剖面法：每个预算 `C` 下扫一串尺寸，用最终 loss 拟合抛物线，**顶点**就是该预算下的
最优点 `N_opt(C)`。再对 `(C, N_opt)` 拟合幂律得到 `a`。

### 3.1 网格怎么定

- 选定 **3~4 个 isoFLOP 预算 `C`**（量级拉开，见下）
- 每个 `C` 从 §2 的 7 档里挑 **4~5 档**尺寸
- token 数由 `D = C/(6N)` 反推；`max_steps` 由 `D` 和每步 token 数算出，
  **cosine 周期必须精确等于 `max_steps`**（`train_abl.py` 已固定这个行为）

### 3.2 ⚠️ 尺寸跨度要靠**参数密度**保证，不能靠感觉

抛物线要能定出顶点，**必须让尺寸跨过最优点两侧**。但"最优点在哪"正是我们要测的——所以
选尺寸时**不能**假设谷底落在某个 token/param 比值上（那等于把答案预先塞进设计里）。

可用的判据是**参数密度**：同一预算 `C` 下，一个尺寸如果 `N` 太小，它会训练得"欠拟合"
（loss 高）；如果 `N` 太大，token 太少也"欠训练"（loss 也高）；中间存在最低点。所以
设计网格时按**参数密度 `C/N³` 由稀到密铺开**（即从"小模型、很多 token"到"大模型、很少
token"），让几何上确实存在一个内部极小值，而不去猜它在哪。

**实操**：每个预算下先跑 4 档，拟合看有没有内部极小。**若单调（没有谷）就补点**——
`train_abl.py` 能随时补单个 run，不必重跑已有的。补点必须基于"几何上还不够宽"这个理由，
不能基于"我猜谷应该在那边"。

### 3.3 ⚠️ 算力口径必须两个都记

`budget.json` / `summary.json` 里同时存了两个口径（`train_abl.py` 自动算）：

```
C_total  = 6·N_total·D      ← N 含 embedding（论文的 C = 6ND 用这个）
C_nonemb = 6·N_nonemb·D     ← 彻底排除 embedding
```

**两者在 14m 上差 11.8 倍**（embedding 占 91.5%），在 1.2b 上只差 1.2 倍。所以"用哪个
口径"会**系统地改变谷底位置**，尤其在小模型端——这是本实验一个真实的不确定性来源，
必须两个都拟合、把差异报告出来，而不是选一个更"好看"的。

（论文把 embedding 算进 N，但它的 vocab 只有 32k 且模型都很大，所以这个选择在它那里
几乎无影响；我们 vocab=50304 且要跑到 14m 这种尺寸，论文的前提在**小端不成立**。）

⚠️ **绝对不要用 HF 的 `total_flos`** —— 它是 `6 × num_parameters(exclude_embeddings=True) × D`，
在 `tie_word_embeddings=False` 时只排除了一个 embedding，既非 `N_total` 也非 `N_nonemb`，
且偏差随尺寸变化（14m 0.542 → 1.2b 0.920）。详见 §9.5。

## 4. 数据（✅ 已就绪）

**`fineweb-edu` 的 `sample/100BT` 子集**，已在机器上：
`/volume/data/hjiang02/open_source/datasets/fineweb/sample/100BT`（**只读，不要动**）。

实测盘点结果（`python scripts/corpus.py`）：

| 项 | 值 |
|---|---|
| 分片 | **150 个 parquet**，282G |
| 行数 | 147,639,585 |
| tokens | **103.03B**（parquet 自带 `token_count` 列口径） |
| 列 | `text` / `id` / `dump` / `url` / `date` / `language` / `language_score` / `token_count` |

**能直接用，不需要转换**，但有两个必须显式指定的字段（否则跑不起来或读到空）：

```json
{"<数据集名>": {"file_name": "raw", "formatting": "alpaca", "columns": {"prompt": "text"}}}
```

- `formatting: alpaca` —— 走非 sharegpt 分支
- **`columns.prompt = text`** —— pt 阶段框架最终读 `_prompt` 列（`pretrain.py:30`），
  而 `columns.prompt` 默认是 `"instruction"`。我们的正文列叫 `text`，不映射就读不到

另外两个设计取舍：
- **不预 tokenize** —— 框架自己 tokenize + packing，省掉一份几百 G 的中间产物
- **用软链接不拷贝** —— 每个 run 的 `data/raw/` 是指向语料分片的软链接。282G 语料
  只有一份，每个 run 复制一份会把配额吃光

**token 数怎么算的**：parquet 的 `token_count` 是 fineweb 自家 tokenizer 的口径，与我们
用的 Pythia tokenizer 不同。抽样 3000 条实测**偏差只有 1.0033 倍（≤0.4%）**，所以直接拿
它做预算，不必为了数 token 把 282G 跑一遍（盘点 150 个分片只要 66 秒）。

**为什么可以直接用这一份、不额外凑 math/code 配比**：本实验测的是"在**固定数据集**上
`N` 与 `D` 的最优配比"，不是"数据集如何影响缩放律"——那是另一个问题。所以只要语料
①在 100B 量级 ②质量大体均匀 ③不重复，我们的对照就在内部自洽：所有尺寸吃的是同一份
数据的**不同切片**（按 run 名做种子，互不重叠），数据集这个变量被控制住了。

⚠️ 这里刻意**不引用**任何"缩放律与数据集无关"之类的现成结论来给这个选择背书——
那是待检验的命题，不该当成设计前提。如果最后各尺寸的 loss 有异常，**"切片之间难度不均"
是必须排查的嫌疑之一**（`runs/*/budget.json` 里记了每档用了哪些分片，可回溯）。

100BT 是**单次 run** 的上限（超过 1 epoch 就要引入重复数据的修正），
多个 run 可以复用同一份语料。

⚠️ **磁盘**：`/volume/data` 配额**长期只剩 <1G**。语料已在盘上（不额外占），且
`train_abl.py` 训练后会自动删掉权重等大件，**单个 run 现在只占 ~0.4M**（实测
`test-prune-14m` 362K，清理掉了 59.9M = 99.7%）。跑网格前仍建议清掉旧 run，
并注意 `1.2b` 档在**训练期间**仍需要足够空间写临时权重（约 4.8G）。

## 5. 实施步骤

### 阶段 0：环境 + 语料 ✅ **已完成**
- ✅ 环境已就绪，无需搭建（见 §8：用平台自带 `/usr/bin/python3`，不要建 conda）
- ✅ 语料已就绪，无需转换（见 §4；接入方式已验证）
- ✅ `scripts/corpus.py` 能盘点/切片/落地数据集
- ⚠️ **仍待解决**：`/volume/data` 配额（见 §4 ⚠️）—— 不阻塞训练，阻塞**存权重**

### 阶段 1：小预算先验证闭环 ✅ **基本完成**
- ✅ `scripts/train_abl.py` 一条命令跑通全流程（备数据→配置→训练→出图）
- ✅ 端到端实测两个尺寸，均 exit 0，**参数量与公式逐位相符**：

  | run | 参数量（实测 vs 公式） | loss | 用时 |
  |---|---|---|---|
  | `pythia-14m` | 14,067,712 == 14,067,712 ✓ | 11.02 → 6.25 | 4.4 min |
  | `pythia-23m` | 22,876,032 == 22,876,032 ✓ | 10.98 → 7.22 | 4.1 min |

- ✅ 参数复核内建（`cross_check_params`），每次 run 自动对公式
- ✅ swanlab 本地模式可用；云端需 `SWANLAB_API_KEY`
- ⏳ **尚未做**：
  - **扫 batch / seq 把 MFU 提上去**。见 §1 已测数据：MFU 天花板 ~25~27%，且
    **seq=2048 时最高（25.4%）并随序列长度单调下降**，所以起点已定为 **seq=2048**；
    再往上调的是**宽度 / batch**，不是注意力实现（三种注意力实现差异只有 1~2%）
  - 用 `pythia-14m` 对齐官方曲线（需要官方权重做对比，见 §2）
**不要跳过。**

### 阶段 2：**按尺寸调 batch / LR / β2**（~1 周）⚠️ 不可跳过

**这是整个计划最容易失败的地方，而失败方式是"悄悄把指数推偏"。** 若不做这一步，各尺寸
会各自处在不同的"调参质量"下，于是 loss 的差异里混进了超参差异，拟合出的指数就不再是
模型规模的函数 —— 而是调参质量的函数。

**这一步的正确做法是自己扫，不要抄论文的数**（理由见文档开头）：对每个尺寸，用最短的
预算短跑几个点，扫出各自的 LR / batch / β2，**以"该尺寸 loss 最低"为唯一判据**。

要调的（每个尺寸都单独扫）：
- **batch size（按 token 计）**
- **max LR**：在各自尺寸上扫 3 个值，取 loss 最低者
- **AdamW β2**：小 batch 下 β2 取大（0.99/0.999）通常更稳；仍以实测为准
- **grad clip**：先用 1.0
- **warmup**：`warmup tokens ≈ N` 是个合理起点，但同样要验
- ⚠️ **初始化**：HF 的 `GPTNeoXPreTrainedModel._init_weights` 对**所有尺寸**都用固定
  `initializer_range=0.02`，**没有** EleutherAI 原实现里 `1/√(2L)` 的残差输出缩放。
  含义：① 大 `L` 档若 loss 一开头就炸，先怀疑 init 而不是自己写错了；② 在 HF 里
  from-scratch 训出来的模型与 pythia 官方权重**不是同一套 init 方案**，交叉验证要留这个偏差

⚠️ **记下每个尺寸最终选了什么**。若最后指数偏了，第一件要查的就是"各尺寸的调参是否同等
充分"——所以调参过程本身也要留证据（`runs/` 里逐 run 都有 `train.yaml`）。

### 阶段 3：跑满网格（随预算）
按 §3 定稿的网格逐个跑。**每跑完一个尺寸就立刻拟合抛物线**，看有没有内部极小；
**若单调就补点**（补点的理由是"几何上跨度不够"，不是"我猜谷在那边"）。

### 阶段 4：拟合与对照（~2 天）
- 每条 isoFLOP 曲线拟合抛物线 → 顶点即 `N_opt(C)`、`D_opt(C)`
  - **两个算力口径各拟合一遍**（`C_total` 与 `C_nonemb`，见 §3.3），报告差异
- 对 `(C, N_opt)` 和 `(C, D_opt)` 拟合幂律 → 得到 `(a, b)`，**给出置信区间**
- 用全部终点 loss 拟合 `L(N,D) = E + A/N^α + B/D^β`
- **到这里才可以翻论文**：对照它的指数、Epoch AI 复现的修正指数
- **若不符，做归因**（这是有价值的产出，不是失败）：
  ① 各尺寸调参是否同等充分（阶段 2 的记录）
  ② 算力跨度是否太窄（§3.2）
  ③ embedding 口径把谷底挪了多少（§3.3）
  ④ 是否超了 1 个 epoch

## 6. 怎么判断"做成了"

**先分清"实验做对了"和"结论是什么"**——这两件事必须分开评价。做对了是过程合格；
结论是什么是测量结果。**不能因为结论不是 0.5 就判定实验失败。**

### 6.1 过程合格（这些是硬要求）

| 检查项 | 要求 | 说明 |
|---|---|---|
| 网格定稿先于开跑 | 必须 | §3 冻结后才开跑，中途不改（补点除外，且理由须是"跨度不够"） |
| 每条 isoFLOP 曲线有内部极小 | 必须 | 单调 = 跨度不够，按 §3.2 补点。**但不许为了凑出谷而反复挪网格** |
| 各尺寸调参同等充分 | 必须 | 阶段 2 的记录可查：每个尺寸都是各自扫出来的最优 |
| 原始 loss 留存 | 必须 | `runs/*/summary.json` + `trainer_log.jsonl` 全留，别人可复算 |
| 报告不确定性 | 必须 | 指数给置信区间；两个算力口径的差异一并报 |

### 6.2 结论（**开放，不预设**）

| 量 | 我们测出来是多少就是多少 | 事后对照用的参考值 |
|---|---|---|
| `a`、`b` | **待测** | 文献里出现过 0.49/0.51、0.35/0.35 等不同值，本身就说明它依赖实验设置 |
| `a + b` | **待测** | 由 `C = 6ND` 约束，理论上是 1 |
| tok/param | **待测** | 文献有 20 的说法 |
| 谷的尖锐程度 | **待测** | 越平 = 该预算下最优尺寸越不确定，这本身是结论 |

⚠️ **若 `a` 落在 0.6~0.9**：文献（Porian et al.）指出这通常意味着各尺寸调参不够充分。
**先查阶段 2 的记录**再下结论——但要把它当作一个**已记录的偏差**正常报告，
而不是"没复现出来所以再调调看"。查询归查询，不许据此重选网格。
**诚实的负面结果也是结果。**

## 7. 目录结构

```
ScalingLaw/
  PLAN.md                       # 本文件
  scripts/
    generate_model_configs.py   # 从模板目录派生尺寸阶梯(config+tokenizer, 无权重)
    corpus.py                   # 语料盘点 / 切片规划 / 落地成 LLaMA-Factory 数据集
    train_abl.py                # 一个 run 的编排: 预算 -> 备数据 -> 生成配置 -> 训练 -> 出图
    plot_run.py                 # 读本地 loss 记录画曲线(单 run / 多 run 叠加)
    # run_grid.py               # (阶段 3 再写) 批量执行 + 断点续跑
    # fit_scaling.py            # (阶段 4 再写) 抛物线拟合 + 幂律拟合
  models/
    templates/pythia-14m/       # 下载的官方模板(尺寸设计的锚点)
    configs/                    # 生成的各尺寸 config(tokenizer 为软链接)
      README.md                 # 尺寸表 + 设计规则(脚本自动生成)
  runs/<run 名>/                # 每次实验的全部产物, 见 §9
  logs/<run 名>.log             # 每次 run 的完整 stdout
  reports/                      # (阶段 4) 拟合结果(带时间戳)
  submodules/LLaMA-Factory/     # 训练框架(submodule, 需 init, 见 §8.1)
```

⚠️ **软链接注意**：`models/configs/*/tokenizer.json` 是指向 `models/templates/` 的软链接，
模板本身（config + tokenizer）已随仓库入库，所以 **clone 下来就能直接用**，不需要额外的下载步骤
（模板下载是一次性操作，刻意没有放进 `generate_model_configs.py`；真要重新下就手动
`HF_TOKEN=... hf download EleutherAI/pythia-14m --local-dir models/templates/pythia-14m`）。
另外 `workspace/LLaMA-Factory -> submodules/LLaMA-Factory` 这条链接（§8.1）需要重建。
每个 run 的 `runs/<名>/data/raw/*.parquet` 也是软链接，指向 open_source 里的语料。

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

## 9. 一次实验的产物说明（`runs/<run 名>/`）

`train_abl.py` 跑完一个 run 后，产物落在 `runs/<run 名>/` 下。**每个 run 自带全部证据**，
事后不用回忆当时怎么跑的。

### 9.1 我们生成的（复现与口径的依据）

| 文件 | 是什么 | 为什么留着 |
|---|---|---|
| `train.yaml` | 完整的 LLaMA-Factory 配置 | **可直接重跑**；也是"当时到底用了什么超参"的唯一权威 |
| `budget.json` | `N_total`/`N_nonemb`/`D`/`C_total`/`C_nonemb`/`tokens_per_step`/`steps`/两种 tok-per-param | **阶段 4 拟合的输入**——横轴 `C` 和纵轴 `loss` 都从这里取 |
| `data/dataset_info.json` | 本次语料的数据集配置（`columns.prompt=text` 等） | 记录数据是怎么接进去的 |
| `data/raw/shard_NNNN.parquet` | **软链接**，指向 `open_source/.../100BT/` 里的真实分片 | 记录"这次到底喂了哪些分片"；不拷贝，282G 语料只存一份 |
| `loss.png` | loss-vs-step + loss-vs-tokens 双面板曲线 | 一眼看收敛形状 |
| `summary.json` | 首/末/尾部均值/最小 loss、`N`/`D`/两种 `C` | 阶段 4 汇总时不用重读 parquet |
| `train.log` | 框架完整 stdout | 出问题时唯一的现场 |

### 9.2 框架自己写的（`out/` 下，**不要手改**）

| 文件 | 是什么 | 备注 |
|---|---|---|
| `out/trainer_log.jsonl` | **逐点 loss/lr/epoch，每步一行（追加写）** | ★ **loss 的唯一必需来源**，`loss.png` 与 `summary.json` 都从它来；上报通道挂了也不丢。追加写 → `output_dir` 必须逐 run 分开 |
| `out/trainer_state.json` | HF 的完整训练状态（含 `log_history`、`grad_norm`、`total_flos`） | 与 jsonl 的 step/loss 相同，但多 `grad_norm`；结构嵌套 |
| `out/train_results.json` | 收尾汇总：`train_loss`/`train_runtime`/`samples_per_second`/`total_flos` | `total_flos` **别用**，见 §9.5 |
| `out/runs/*/events.out.tfevents.*` | TensorBoard 事件（`report_to: tensorboard`） | 本地 loss 的第二份记录 |
| `out/swanlab_public_config.json` | 指向 `runs/_swanlab/run-...` 的**指针** | 关联键是 `run_id` |

**被删掉的**（见 §9.3）：`model.safetensors`、`*.bin`、`tokenizer.json`、`config.json`、
`generation_config.json`、`chat_template.jinja`、`README.md`、`all_results.json`、`checkpoint-*/`。

### 9.3 训练结束后会自动删掉权重（run 目录只剩 ~250K）

LlaMA-Factory **没有任何开关能跳过存权重**（`pt/workflow.py:64` 的 `trainer.save_model()`
是无条件调用，且未 override HF 原生实现）。而本实验只需要 **loss-vs-C**，权重完全用不上，
14m 就要 54M、1.2b 要 4.8G —— 配额只剩几百 M，不删必然撑爆。

所以 `train_abl.prune_run_artifacts()` 在**出图之后**清掉与缩放律无关的大件：

| 删掉 | 为什么不需要 |
|---|---|
| `model.safetensors`、`*.bin` | 权重，拟合用不到（要留就加 `--keep-model`） |
| `tokenizer.json`（3.5M） | 分词器副本，`models/configs/` 里已有 |
| `config.json`、`generation_config.json`、`chat_template.jinja` | 被框架改写过的副本；原始 config 在 `models/configs/` |
| `README.md` | 自动生成的模型卡，无手写内容 |
| `all_results.json` | `train_results.json` 的孪生副本（md5 相同） |
| `checkpoint-*/` | 中间检查点（默认本来就不存） |

| 保留 | 为什么必须留 |
|---|---|
| `trainer_log.jsonl` | **loss 的唯一必需来源** |
| `trainer_state.json` | 含 `grad_norm`（jsonl 没有），排查训练异常要看 |
| `train_results.json` | 收尾汇总 |
| `out/runs/*/events.out.tfevents.*` | TensorBoard 记录，只有 ~30K |
| `train.yaml` / `budget.json` / `loss.png` / `summary.json` / `train.log` | 见下 |

**实测**（`runs/test-prune-14m`）：删 9 项释放 **59.9M**（占 99.7%），run 目录从 60M 降到
**362K**。权重只在**训练期间**占盘，跑完即释放。

### 9.4 swanlab

`runs/_swanlab/` 是 **local 模式的落盘目录，被所有 run 共享**（不是每个 run 一份）：

- `run-<时间戳>-<id>/` —— 每个 run 一个子目录，含 `files/config.yaml`、`console/`、`logs/`
- `out/swanlab_public_config.json` —— 由框架写在 run 自己的目录里，指明本次 run 对应
  `_swanlab/` 下的哪个子目录（`run_id` 就是关联键）
- 云端模式下 `_swanlab/` 仍会生成一份本地缓存，作为上传失败的兜底

### 9.5 ⚠️ 算力口径：**全部由我们自己算**，不信任库给的数

**坑 1：`tokens_per_step` 必须乘 `world_size`。** 漏掉 GPU 数会让实际 `D` 只有预期的
1/4，isoFLOP 剖面直接废掉。已在 `train_abl.plan_budget` 里固定。

**坑 2：HF 的 `total_flos` 不能当成 `C = 6ND` 用，所以干脆不用它。**
实测（transformers 5.8.0，`trainer.py:3900`）HF 的口径是：

```
6 × num_parameters(exclude_embeddings=True) × tokens
```

而 `exclude_embeddings=True` 在 `tie_word_embeddings=False` 时**只排除了输入
embedding，LM head 仍被算进去**。实测：

| 量 | 值 | 是什么 |
|---|---|---|
| `num_parameters()` | 14,067,712 | = `N_total`，与我们的公式一致 ✓ |
| `num_parameters(exclude_embeddings=True)` | 7,628,800 | = `N_total − vocab·d`，**中间量** |
| 我们算的 `N_nonemb` | 1,189,888 | 真正排除两个 embedding |

所以 HF 那个数**既非 `C_total` 也非 `C_nonemb`**。危害在于偏差随尺寸变化：

| 尺寸 | HF 口径 / `C_total` |
|---|---|
| `pythia-14m` | **0.542** |
| `pythia-86m` | 0.701 |
| `pythia-442m` | 0.862 |
| `pythia-1.2b` | **0.920** |

从 14m 到 1.2b 差 **1.7 倍**。若拿它当横轴 `C`，小模型会被系统性压向一侧，**直接扭曲
拟合出的指数 `a`** —— 且因为是尺寸相关的偏差、不是常数偏移，很难事后发现。

**对策**：`train_abl.param_counts()` 直接按公式从 `config.json` 拆参数
（`N_非emb = 12Ld² + 13Ld + 2d`，`N_总 = N_非emb + 2·vocab·d`），
`plan_budget()` 据此算出**两个**算力口径写进 `budget.json`：

```
flops        = 6·N_total·D     ← 主口径（论文把 embedding 算进 N）
flops_nonemb = 6·N_nonemb·D    ← 敏感性分析口径（彻底排除 embedding）
```

`log_flops_calibers()` 每次 run 打印两者比值，并**对照** HF 的 `total_flos` 提醒它
落在哪儿（实测比值与预测 `1 − vocab·d/N_total` 完全吻合）。`plot_run.summarize()` 也只从
`budget.json` 取口径，绝不读 `train_results.json` 的 `total_flos`。

**所以横轴一律用 `budget.json` 里的 `flops`（= `6·N_total·D`，论文口径）。**
`train_abl.warn_flos_trap` 每次 run 都会把两者的比值打出来提醒（实测比值与预测
`1 - vocab·d/N_total` 完全吻合）。注意这与论文"把 embedding 算进总参数"的表述一致
（§3），口径不要混。

## 10. 关键参考

- 原文：arXiv:2203.15556（Table A9 尺寸表、App. B cosine、App. C 跨数据集、App. F FLOPs 计数）
- Epoch AI 复现：arXiv:2404.10102 —— 重拟合得 α≈β≈0.35（修正指数与 20 tok/param 自洽）
- Porian et al. 2024：arXiv:2406.19146 —— 调参对指数测量的影响（阶段 2 的依据）
- datablations：arXiv:2305.16264 —— 重复数据下的缩放律修正（若超 1 epoch 必读）
- `epoch-research/analyzing-chinchilla`：从论文图里像素级反推的 245 个数据点
- Pythia：arXiv:2304.01373 —— 模板来源，也是交叉验证的基准曲线
