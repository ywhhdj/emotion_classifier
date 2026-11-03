# v1 → v2 落地对照

本文记录代码改造的**文件级对照**，配合 `docs/改造设计_v2.md` 阅读。

## 一、为什么换基座

| | v1 | v2 |
|---|---|---|
| 编码器 | `paraphrase-multilingual-MiniLM-L12-v2`（384 维句向量） | `xlm-roberta-base`（768 维） |
| 训练目标 | 语义相似度对比学习 | MLM（100 种语言） |
| 副作用 | 句向量**主动抹掉**与"同义"无关的信息 | 保留 token 级交互 |
| 症状 | 多语言 <70%；"I happily lowered my head" 判成害羞 | 症状即病因，换基座后消除 |

数据事实（`data/emotion.csv`，2556 行）：zh 1863 / en 356 / ja 337；
文本均值 9.15 字符、max 45 —— **极短文本，无长文本场景**。

## 二、文件级改动

### 训练侧

| 文件 | 改动 |
|---|---|
| `train/config.py` | `MODEL_NAME=xlm-roberta-base`、`INPUT_DIM=768`、`MAX_SEQ_LEN=48`；新增 LoRA / prompt / 回译 / 跨语言扩充 / 裁剪剪枝 / 蒸馏配置段 |
| `train/models.py` | 新增 `build_encoder_lora`、`merge_lora`、`build_param_groups`、`XlmrTorchEncoder`、`PromptMaskEncoder`、`TextClassificationDataset`、`make_collate`、`_xlmr_position_ids`；`ONNXEncoder` 语义改 XLM-R；`ChunkedEncoder` 标 deprecated |
| `train/train.py` | 新增 `stage_train_e2e`（文本级端到端）、`stage_calibrate`；`stage_onnx` 改为整图导出；`stage_embed` 保留给 cached/prompt；`cross_validate` 支持 e2e |
| `train/augment.py` | EDA 按语言降级：`LANG_SAFE_EDA=("en",)`；`punctuation` 默认关闭（`。`→`！` 会把平静改判成生气） |
| `train/backtranslate.py` | 新增：NLLB-200 回译 + 三重质量闸门 + `bt_cache.jsonl` |
| `train/crosslingual.py` | 新增：zh/en → ja/en 翻译扩充，`group_id` 防泄漏，`is_mt` 样本降权 |
| `train/cli.py` | 新增 `--train-mode/--lora-*/--use-bt/--cross-lingual/--max-seq-len/--use-prompt/--calibrate`；cached 模式自动套用旧的 epochs/lr/batch |

### 推理侧（`src/`，对外 API 零变化）

| 文件 | 改动 |
|---|---|
| `src/models/config.py` | `EXPECTED_DIM=768`、`ENCODER_TOKENIZER=xlm-roberta-base`、`VERSION=0.2.0`；去掉 sentence-transformers 下载分支；新增 `is_whole_graph()`、`find_calib()` |
| `src/models/encoder.py` | `ONNXEncoder` 改 XLM-R（pad_id=1、无 token_type_ids）；新增 `RustEngineEncoder`；`SentenceTransformerEncoder` 标 deprecated |
| `src/predictor.py` | 支持整图单 session；Rust → ONNX → PyTorch 后端链；温度/per-class 校准；补上 `predict_label`（README 承诺过但 v1 未实现） |
| `src/runtime.py` ★ | Rust 引擎单例与后端选择 |
| `src/languages.py` ★ | 语言归一化与规则检测 |
| `src/threshold.py` ★ | `calib.json` 加载与应用 |
| `src/exceptions.py` ★ | 异常层级 |
| `src/legacy.py` ★ | v1 兼容外壳（自动识别 384 维旧产物） |
| `src/models/pytorch.py` | `input_dim` 默认 384 → 768 |

### 新增目录

- `rust/` — 6 个 crate：`emotion-core`（类型/配置）、`emotion-tokenizer`（SentencePiece+归一化+语言检测）、`emotion-runtime`（candle/ort 后端 + 批调度）、`emotion-postprocess`（softmax/温度/阈值/top-k）、`emotion-python`（PyO3 → `src._core`）、`emotion-cli`（自检/bench）
- `scripts/` — `vocab/`（建语料/裁剪/校验）、`prune/`（剪枝+蒸馏）、`calibrate/`、`export/`、`bench/`、`ci/`（分词一致性）
- `configs/` — `gc.json` / `tc.json` / `vc.json`
- `tests/test_core.py` — 纯逻辑单测（不下载模型）

## 三、训练目录 vs 部署目录（v2 明确分离）

v2 之前这两件事是混在一起的，现在拆开：

| | 训练侧 | 部署侧 |
|---|---|---|
| 目录 | `models/xlmr-zh-en-ja/`、`model/best/`、`model/pruned/` | `models/deploy/` |
| 格式 | HuggingFace 目录（`config.json` + `model.safetensors` + `sentencepiece.bpe.model`） | ONNX 整图 + tokenizer + `label_map.json` + `calib.json` |
| 使用者 | 只有 `train/` | `src/` 与 Rust 内核 |
| 产出方式 | `trim_vocab.py` → 微调 → `prune_encoder.py` | `scripts/export/package_deploy.py` |

**ONNX 是导出产物，不是训练输入。** 训练必须 `from_pretrained(HF 目录)`；
`src/` 不认识训练目录，只加载部署目录。

### 基座解析逻辑（`train/config.py`）

```
Config.BASE_MODEL_DIR = "models/xlmr-zh-en-ja"   # 三语裁剪产物
Config.HUB_MODEL_NAME = "xlm-roberta-base"       # 仅回退用
Config.is_complete_hf_dir(d)   # 三件套齐全才算「就绪」
Config.encoder_dir()           # 本地裁剪产物优先，否则回退 hub
Config.is_trimmed()            # 当前基座是否已裁剪
```

`is_complete_hf_dir` 要求 `config.json` + 权重 + 分词模型**三者齐全**，
避免半截产物（比如只跑了裁剪没写 spm）被当成基座静默使用。

### 新增文件

- `scripts/export/package_deploy.py` —— 训练产物 → 自包含部署目录，
  产出 `deploy_manifest.json`（来源 / vocab / 量化 / 校验和），结尾跑 smoke test。
- `emotion_classifier/__init__.py` —— 安装后的对外包名，`src` 的薄别名。

### src 侧不再联网

`src/predictor.py::__init_tokenizer` 只在部署目录找 tokenizer，
找不到就抛 `FileNotFoundError` 并提示跑 `package_deploy.py`，
**不会静默回退到 hub 下载**（v2 之前会，离线环境直接跪）。

---

## 四、两个必须注意的 XLM-R 坑

1. **position_ids 从 `padding_idx + 1 = 2` 起算**
   HF 用 `create_position_ids_from_input_ids` 按非 pad token 累计生成。
   Rust 侧 `emotion-runtime::create_position_ids` 与 Python 侧
   `train.models._xlmr_position_ids` 都复现了该逻辑（`tests/test_core.py` 有单测）。
   不复现的话，右 padding 的短句位置编码全错，精度掉 3–10 pt。

2. **LoRA 的 `target_modules` 是 `query/key/value`**
   HF 的 `XLMRobertaSelfAttention` **不是** `q_proj/v_proj`。写错会静默退化成
   "零个可训练模块"——不报错，但编码器完全没有适配。

## 五、词表裁剪的三条路径

| 路径 | 工具 | 结论 |
|---|---|---|
| P1 主路径 | `textpruner.VocabularyPruner` | 官方对 XLM-R 有一等支持（XNLI 示例即 XLM-R） |
| P2 对照 | hf-trim | **原生不支持 XLM-R**（仅 BART/mBART/T5/mT5）；需自研 `XLMRTrimmer`；且不支持 fast tokenizer，XLM-R 底层是 SentencePiece，裁剪后必须同步重写 `.model`，否则静默产生已删除的 id |
| P3 产物路径（推荐） | 新训三语 SentencePiece 取交集 | 交集 token 的 embedding 行直接从 `xlm-roberta-base` 复制，产物自带自洽 `.model`，HF 与 Rust 的 `sentencepiece` crate 都能直接加载，无 ID 错位风险 |

脚本：`scripts/vocab/trim_vocab.py --mode {intersect|textpruner|hf-trim}`，
裁剪后必须跑 `scripts/vocab/check_vocab.py`（7 项校验，CI 门禁）。

## 六、执行顺序（不能颠倒）

```
词表三语裁剪 → DAPT(MLM) → 微调 → LoRA merge → 剪枝 → 蒸馏恢复 → 校准 → 导出
```

- 词表裁剪必须在微调**之前**（否则被删的行白训）；
- LoRA merge 必须在剪枝**之前**（剪枝工具需要看到完整权重）；
- 剪枝必须在蒸馏**之前**（剪完结构损伤，蒸馏恢复）。

## 七、实验矩阵

| # | 实验 | 决策 |
|---|---|---|
| E0 | 复现 v1（MiniLM + 头） | 得到 `F1_minilm` |
| E1 | 路径 B：XLM-R 冻结 + prompt | 应显著 > E0，验证换基座生效 |
| E2 | 路径 A：XLM-R + LoRA | 相对 E1 ≥ +1.5 pt 则选 A |
| E3 | + 回译 | 看 en/ja 提升 |
| E4 | + 跨语言扩充 | 看语言极差是否收窄到 ≤5 pt |
| E5 | + 三语词表裁剪 + DAPT | 掉点 ≤1 pt 采纳 |
| E6 | + 剪枝 + 蒸馏 | 掉点 ≤1 pt 采纳 |
| E7 | + 校准 | 一般 +0.5~1.5 pt |
| E8 | fp16 / int8 | 掉点 ≤0.5 pt 放行 |
| E9 | Rust 内核一致性 | logits <1e-4、分词零差异才切换 |

每个实验记录 `(vocab, heads, ffn, quant, seed)`，3 seed 取均值 ± std，
方差 >1 pt 的方案不采纳。
