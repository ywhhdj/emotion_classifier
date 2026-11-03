# Emotion Classifier

多语言轻量情感分类模型包，支持 **18 种情感标签**，覆盖 **中文 / 英文 / 日文**。

- **v2.0**：编码器由 `paraphrase-multilingual-MiniLM-L12-v2`（语义相似度句向量）换为 **`xlm-roberta-base`**，支持 **LoRA 端到端微调**；
- 轻量化：三语词表裁剪（250k → 32k~48k）+ TextPruner 编码器剪枝 + FP16/INT8，交付 **158 MB(fp16) / 79 MB(int8)**；
- 推理内核为 **Rust**（分词 / 批调度 / 前向 / 后处理），`src/` 对外仍是 **Python 胶水层**，API 与 v1 完全兼容。

> v1 的病因：MiniLM 是**语义相似度**模型，句向量会主动抹掉与"同义"无关的信息，
> 而这恰恰是情感分类最需要的（程度副词、否定、情绪词的细微差别）。
> README 里自述的两个症状（多语言准确率 <70%、"I happily lowered my head" 判成害羞）都源于此。

---

## 1. 安装

### 仅推理

```bash
pip install .                      # 含 Rust 内核（需要 Rust 工具链）
EMOTION_BACKEND=onnx pip install .  # 纯 Python 回退（无 Rust 工具链时）
```

### 训练

```bash
pip install ".[train]"        # torch + transformers + peft
pip install ".[full]"         # + 剪枝(textpruner) + 回译(sentencepiece) + v1 兼容
```

> **Python 版本**：`>= 3.10`（代码使用 PEP 604 的 `str | Path` 与内置泛型）。

---

## 2. 快速开始（API 与 v1 完全一致）

```python
from emotion_classifier import EmotionClassifier

clf = EmotionClassifier(auto_download=True)

# 单条预测
print(clf.predict("脸颊泛红，偷偷瞄了你一眼", top_k=3))
# [[('害羞', 0.93), ('疑惑', 0.04), ('难为情', 0.02)]]

# 批量
texts = ["i'm so excited i can hardly breathe",
         "怖くて震えが止まらない",
         "气鼓鼓地跺着脚瞪着你"]
for text, preds in zip(texts, clf.predict(texts, top_k=2)):
    print(text, "→", preds)

# 只要最可能的 1 个标签
print(clf.predict_label("今天的天气真好"))

# 概率矩阵（评估 / 阈值调优）
probs = clf.predict_proba(texts)          # (N, 18)

# 指定语言（不传则自动检测）
clf.predict("今日はとても嬉しいです", lang="ja")

print(clf.get_labels())
```

从源码仓库直接使用（未安装时）：

```python
import sys; sys.path.insert(0, "/path/to/emotion_classifier")
from src import EmotionClassifier
```

### 构造参数（新增项标 ★）

| 参数 | 默认 | 说明 |
|------|------|------|
| `model_dir` | `~/.emotion_classifier/models` | 模型目录；环境变量 `EMOTION_MODEL_DIR` |
| `backend` | `"auto"` | `auto` / `rust` ★ / `onnx` / `pytorch` |
| `auto_download` | `False` | 缺失模型时是否自动下载 |
| `device` | 自动探测 | `cpu` / `cuda` |
| `tokenizer_dir` | `<model_dir>/tokenizer` | 本地 tokenizer 目录 |
| `encoder_path` | `None` | 自定义编码器（仅旧两段式产物需要） |
| `encoder_cache_size` | `0` | 向量缓存条数；交互场景建议 1024 |
| `num_threads` ★ | `4` | Rust 内核线程数 |
| `max_batch` ★ | `32` | 内核批调度上限 |
| `use_lang_token` ★ | `True` | 输入前拼 `<zh>/<en>/<ja>` |

### 上下文管理器

```python
with EmotionClassifier() as clf:
    results = clf.predict(["text1", "text2"], top_k=5)
```

### 18 种情感标签

| 序号 | 标签 | 说明 | 序号 | 标签 | 说明 |
|:--:|------|------|:--:|------|------|
| 1 | 高兴 | 开心、喜悦、激动 | 10 | 担心 | 焦虑、不安 |
| 2 | 厌恶 | 反感、恶心 | 11 | 无奈 | 无力、放弃 |
| 3 | 哭泣 | 悲伤哭泣 | 12 | 生气 | 愤怒、恼火 |
| 4 | 害怕 | 恐惧、惊慌 | 13 | 疑惑 | 困惑、不解 |
| 5 | 害羞 | 羞涩、脸红 | 14 | 紧张 | 局促、不安 |
| 6 | 平静 | 冷静、安宁 | 15 | 自信 | 确信、从容 |
| 7 | 心动 | 浪漫、喜欢 | 16 | 认真 | 专注、严肃 |
| 8 | 惊讶 | 吃惊、意外 | 17 | 调皮 | 顽皮、戏弄 |
| 9 | 慌张 | 慌乱、手忙脚乱 | 18 | 难为情 | 尴尬、羞耻 |

> 标签 ID 由 `LabelEncoder` 按**字典序**生成（`label_map.json` 中依次为 厌恶、哭泣、害怕、害羞、平静、心动、惊讶、慌张、担心、无奈、生气、疑惑、紧张、自信、认真、调皮、难为情、高兴）。**以训练产出的 `label_map.json` 为准**，不要硬编码 ID。

---

## 3. 命令行用法

```bash
emotion-classify "脸颊泛红，偷偷瞄了你一眼"
emotion-classify "气鼓鼓地跺着脚瞪着你" --top-k 5
emotion-classify --batch "text1" "text2"
emotion-classify --file input.txt
emotion-classify "今天的天气真好" --json
emotion-classify --interactive --cache 1024

emotion-classify "test" --backend rust        # 强制 Rust 内核
emotion-classify "test" --model-dir /path/to/models
emotion-classify "test" --no-download

# 运维
emotion-classify --labels
emotion-classify --info          # 后端 / 编码器 / 整图标志 / 是否校准
emotion-classify --list-models
emotion-classify --update
```

---

## 4. 训练自己的模型

### 数据格式

CSV，至少包含 `text` 与 `label` 两列；可选 `lang` 列（用于语言过滤与分层）。

```csv
text,label,lang
今天心情真好,高兴,zh
i'm so excited i can hardly breathe,高兴,en
怖くて震えが止まらない,害怕,ja
```

当前 `data/emotion.csv`：2556 行，zh 1863 / en 356 / ja 337，**en、ja 每类仅 16~36 条** —— 这是精度的主要瓶颈。

### 两条训练路径

| 路径 | 命令 | 说明 |
|---|---|---|
| **A：LoRA 端到端（推荐）** | `--train-mode e2e` | XLM-R + LoRA(r=8，后 4 层)，文本级训练 |
| **B：冻结编码器 + prompt 完形** | `--train-mode cached --use-prompt` | 编码器冻结，只训分类头；复用预编码缓存，1 小时内出数 |

**先跑 B（成本极低，作为换基座的 sanity check），再跑 A；两者 CV 差距 <1.5 pt 说明瓶颈在数据，算力应投到回译与跨语言扩充。**

```bash
# 路径 B：XLM-R 冻结 + prompt 完形（先出数）
python main.py --train-mode cached --use-prompt \
  --hidden-dim 256 --dropout 0.4 --lr 2e-3 \
  --class-weight effective --focal-gamma 2.0 --metric macro_f1

# 路径 A：XLM-R + LoRA 端到端
python main.py --train-mode e2e \
  --lora-r 8 --lora-unfreeze-from 8 --lr 1e-4 \
  --epochs 20 --early-stop 6 --batch-size 32 \
  --max-seq-len 48 --metric macro_f1

# 回译 + 跨语言扩充（en/ja 数据太少的解法）
python main.py --train-mode e2e --use-bt --bt-factor 1 --cross-lingual

# 5 折交叉验证（小数据集必须用）
python main.py --train-mode e2e --folds 5 --stratify-lang
```

### 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--train-mode` | `e2e` | `e2e` / `cached` |
| `--model-name` | `xlm-roberta-base` | 编码器基座 |
| `--lora-r` | `8` | LoRA rank |
| `--lora-unfreeze-from` | `8` | 从第几层开始挂 LoRA（前层冻结） |
| `--max-seq-len` | `48` | 数据 max=45 字符，48 足够 |
| `--use-lang-token` | `True` | 输入前加 `<zh>/<en>/<ja>` |
| `--use-prompt` | `False` | prompt 完形（仅 cached 模式） |
| `--use-bt` | `False` | 回译增强（NLLB-200） |
| `--bt-factor` | `1` | 每条产出的回译样本数 |
| `--cross-lingual` | `False` | 跨语言扩充（zh/en → ja/en） |
| `--folds` | `0` | 交叉验证折数 |
| `--stratify-lang` | `False` | 按 `标签×语言` 分层划分 |
| `--calibrate` | `True` | 温度标定 |
| `--lang` | 空 | 只保留指定语言，如 `zh,en` |
| `--hidden-dim` / `--dropout` / `--lr` / `--epochs` / `--early-stop` | 见 `--help` | 沿用 v1 |
| `--class-weight` / `--focal-gamma` / `--label-smoothing` / `--mixup-alpha` | 沿用 v1 | 沿用 v1 |
| `--chunk-size` / `--chunk-overlap` | - | **[DEPRECATED]** 长文本分块，v2 不再需要 |

---

## 5. 轻量化流水线

顺序不可颠倒：

```
词表三语裁剪 → DAPT(MLM) → 微调 → LoRA merge → 剪枝 → 蒸馏恢复 → 校准 → 导出
```

```bash
# 1) 构建三语语料 + 裁剪词表（250k → 40k）
python scripts/vocab/build_corpus.py --data data/emotion.csv --out data/corpus/zh_en_ja.txt
python scripts/vocab/trim_vocab.py --mode intersect --target-vocab 40000 \
    --corpus data/corpus/zh_en_ja.txt --out models/xlmr-zh-en-ja
python scripts/vocab/check_vocab.py --model-dir models/xlmr-zh-en-ja     # CI 必跑

# 2) 剪枝（8 头 / FFN 1536）+ 蒸馏恢复
python scripts/prune/prune_encoder.py --model-dir model/best \
    --configs configs/gc.json configs/tc.json --out model/pruned
python scripts/prune/prune_encoder.py --distill --model-dir model/pruned \
    --teacher model/best --out model/distilled

# 3) 校准 + 整图导出 + 压测
python scripts/calibrate/calibrate.py --model-dir model/distilled --out model/calib.json

# 组装自包含部署目录（src 与 Rust 只认这个）
python scripts/export/package_deploy.py --model-dir model/distilled --out models/deploy

python scripts/bench/bench.py --model-dir models/deploy --backend rust -n 10000
```

### 档位

| 配置 | vocab | heads | FFN | 总参数 | fp16 | int8 |
|---|---|---|---|---|---|---|
| XLM-R 基线 | 250002 | 12 | 3072 | 278.0 M | 556 MB | 278 MB |
| **A（推荐）** | 40000 | 8 | 1536 | **78.9 M** | **158 MB** | **79 MB** |
| B（激进） | 32000 | 8 | 1024 | 63.4 M | 127 MB | 63 MB |
| C（保守） | 48000 | 10 | 2048 | 99.3 M | 199 MB | 99 MB |

掉点 > 2 pt 就从 A 回退到 C。

---

## 6. 两类模型目录（**不要混用**）

| | 训练侧 | 部署侧 |
|---|---|---|
| **目录** | `models/xlmr-zh-en-ja/`、`model/best/`、`model/pruned/` | `models/deploy/` |
| **格式** | HuggingFace 目录（`config.json` + `model.safetensors` + `sentencepiece.bpe.model`） | 自包含产物（ONNX 整图 + tokenizer + `label_map.json` + `calib.json`） |
| **谁用** | 只有 `train/` | `src/` 与 Rust 内核**只认这个** |
| **怎么来** | `trim_vocab.py` → 微调 → `prune_encoder.py` | `scripts/export/package_deploy.py` |

> ONNX 是**导出后的产物**，不是训练输入。训练必须从 HF 权重目录 `from_pretrained`；
> 反过来，`src/` 推理包不认识训练目录，只加载部署目录。

```bash
# 训练侧：三语裁剪基座（HF 格式）
python scripts/vocab/trim_vocab.py --mode intersect --target-vocab 40000 \
    --corpus data/corpus/zh_en_ja.txt --out models/xlmr-zh-en-ja

# 训练（自动识别：本地裁剪产物优先，缺失才回退 hub 下载原始 XLM-R）
python main.py --train-mode e2e --base-model-dir models/xlmr-zh-en-ja

# 部署侧：组装自包含目录
python scripts/export/package_deploy.py \
    --model-dir model/pruned --out models/deploy

# 推理：只认部署目录
EMOTION_MODEL_DIR=models/deploy emotion-classify "今天心情真好"
```

`models/deploy/` 产物树：

```text
models/deploy/
├── config.json                    # XLM-R 结构（candle 后端用）
├── model.safetensors              # 权重（candle 后端用）
├── emotion_classifier.onnx        # 整图 FP32
├── emotion_classifier_fp16.onnx
├── emotion_classifier_int8.onnx
├── sentencepiece.bpe.model        # Rust tokenizer 直接在根目录找
├── tokenizer/                     # HF tokenizer（Python 回退路径用）
├── label_map.json
├── calib.json
└── deploy_manifest.json           # 来源 / vocab / 量化 / 校验和
```

**推理侧绝不联网**：`src/` 只在部署目录里找 tokenizer，找不到就报错提示跑
`package_deploy.py`，不会静默回退到 hub 下载。

### import 方式

```python
from emotion_classifier import EmotionClassifier   # 安装后推荐
from src import EmotionClassifier                  # 仓库内开发等价
```

---

## 7. 项目结构

```text
emotion_classifier/
├── src/                          # 推理包（安装后名为 emotion_classifier；纯 Python 胶水层）
│   ├── models/
│   │   ├── config.py             # 模型仓库、校验和、标签/校准文件路径解析
│   │   ├── encoder.py            # ONNX / Rust 编码器（EXPECTED_DIM=768）
│   │   └── pytorch.py            # PyTorch 分类头
│   ├── __init__.py  cli.py  downloader.py  predictor.py
│   ├── runtime.py                # Rust 引擎单例与后端选择 ★
│   ├── languages.py  threshold.py  exceptions.py  legacy.py ★
│   └── data/label_map.json
├── train/
│   ├── config.py                 # MODEL_NAME=xlm-roberta-base, INPUT_DIM=768 ★
│   ├── models.py                 # + build_encoder_lora / PromptMaskEncoder / TextClassificationDataset ★
│   ├── train.py                  # + stage_train_e2e / stage_calibrate / 整图导出 ★
│   ├── augment.py                # EDA 按语言降级（中日文不再被破坏）★
│   ├── backtranslate.py ★         # 回译增强（NLLB-200）
│   ├── crosslingual.py ★          # 跨语言扩充（zh/en → ja/en）
│   ├── cli.py  utils.py
├── rust/                         # 内核（PyO3 → src._core）★
│   └── crates/{emotion-core,emotion-tokenizer,emotion-runtime,
│               emotion-postprocess,emotion-python,emotion-cli}
├── scripts/                      # vocab/ prune/ calibrate/ export/ bench/ ci ★
├── configs/                      # gc.json / tc.json / vc.json ★
├── scripts/
│   ├── vocab/    trim_vocab.py  check_vocab.py  build_corpus.py
│   ├── prune/    prune_encoder.py（剪枝 + 蒸馏）
│   ├── export/   export_onnx.py  package_deploy.py ★
│   └── calibrate/ bench/ ci/
├── emotion_classifier/           # 安装后的对外包名（src 的薄别名）★
├── models/
│   ├── xlmr-zh-en-ja/            # 训练基座（HF，三语裁剪）
│   └── deploy/                   # 部署产物（自包含）
├── model/                        # 训练产物中转（.pt / best / pruned）
├── tests/  data/  docs/  main.py  setup.py  pyproject.toml
```

---

## 8. 模型文件

| 文件 | 说明 |
|------|------|
| `emotion_classifier.onnx` | **整图 FP32**（encoder+head；推荐部署用） |
| `emotion_classifier_fp16.onnx` | 整图 FP16（体积减半；CPU 上回退为 FP32 计算） |
| `emotion_classifier_int8.onnx` | 整图 INT8 动态量化 |
| `calib.json` | 温度 + per-class 阈值（可选） |
| `label_map.json` | 标签映射 |
| `tokenizer/` | XLM-R tokenizer（含 `sentencepiece.bpe.model`） |

加载优先级：`emotion_classifier.onnx` → `*_fp16.onnx` → `*_int8.onnx` → 其他。

---

## 9. 已知限制

1. **en / ja 数据稀疏**：每类仅 16~36 条，任何模型都难学出判别边界。解法是回译（`--use-bt`）与跨语言扩充（`--cross-lingual`），而不是继续调模型。
2. **FP16 在 CPU 上不加速**：ONNXRuntime CPU 会把 FP16 回退为 FP32 计算，收益只在体积；真正加速靠 Rust 内核（candle）或 GPU EP。
3. **INT8 掉点**：embedding 量化是中日语掉点的主要来源，量化后必须重跑三语评测，掉点 > 0.5 pt 就退回 FP16。
4. **回译依赖 NLLB-200**：首次运行需下载 `facebook/nllb-200-distilled-600M`（约 2.4 GB），结果缓存到 `data/bt_cache.jsonl`，只跑一次。
5. **Rust 内核未编译时**：设置 `EMOTION_BACKEND=onnx` 回退到 ONNX Runtime，API 完全一致。

---

## 10. 集成示例

### LangChain Tool

```python
from langchain.tools import BaseTool
from emotion_classifier import EmotionClassifier

class EmotionAnalysisTool(BaseTool):
    name = "emotion_analysis"
    description = "分析文本的情感，返回 top-3 情感标签及置信度"

    def __init__(self):
        super().__init__()
        self.clf = EmotionClassifier(auto_download=True)

    def _run(self, query: str) -> str:
        return str(self.clf.predict(query, top_k=3)[0])
```

### FastAPI 服务

```python
from fastapi import FastAPI
from pydantic import BaseModel
from emotion_classifier import EmotionClassifier

app = FastAPI()
clf = EmotionClassifier(auto_download=True, encoder_cache_size=4096)

class Request(BaseModel):
    text: str
    top_k: int = 3

@app.post("/predict")
def predict(req: Request):
    return {"results": clf.predict(req.text, req.top_k)[0]}
```

---

## License

MIT

## 致谢

- [xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) — 多语言编码器基座
- [TextPruner](https://github.com/airaria/TextPruner) — 词表裁剪与编码器剪枝
- [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) — 回译与跨语言扩充
- [PEFT](https://github.com/huggingface/peft) — LoRA
- [candle](https://github.com/huggingface/candle) / [PyO3](https://pyo3.rs) — Rust 内核与 Python 绑定
