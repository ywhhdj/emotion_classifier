# Emotion Classifier

多语言轻量情感分类模型包，支持 **18 种情感标签**，覆盖 **中文 / 英文 / 日文**。

- 编码器为 `paraphrase-multilingual-MiniLM-L12-v2`（推荐量化版 `model_quint8_avx2.onnx`），分类头是一个 **384 → 128 → 18** 的轻量残差 MLP，可导出 ONNX / INT8 部署。
---

## 1. 安装

### 仅推理（推荐，无需 PyTorch）

```bash
pip install .
```

### 训练（需要 PyTorch）

```bash
pip install ".[full]"
```

> **编码器模型**：首次运行会自动从 HuggingFace 下载 `model_quint8_avx2.onnx`（约 90MB）与 tokenizer，缓存到模型目录。离线环境请手动放置到 `--model-dir` 指定的目录，并加 `--no-download`。

> **Python 版本**：`>= 3.10`（代码使用 PEP 604 的 `str | Path` 与内置泛型，3.8/3.9 无法运行；原 `setup.py` 声明的 `>=3.8` 已在本次修复中更正）。

---

## 2. 快速开始（Python API）

```python
from emotion_classifier import EmotionClassifier

# 初始化（自动检测 ONNX / PyTorch 后端，自动下载模型）
clf = EmotionClassifier(auto_download=True)

# 单条预测
print(clf.predict("脸颊泛红，偷偷瞄了你一眼", top_k=3))


# 批量预测
texts = [
  "i'm so excited i can hardly breathe",
  "怖くて震えが止まらない",
  "气鼓鼓地跺着脚瞪着你",
]
for text, preds in zip(texts, clf.predict(texts, top_k=2)):
  print(text, "→", preds)

# 只要最可能的 1 个标签
print(clf.predict_label("今天的天气真好"))   # [('平静', 0.4219)]

# 只要概率矩阵（用于评估 / 阈值调优）
probs = clf.predict_proba(texts)             # shape (N, 18)

# 查看所有标签
print(clf.get_labels())
```

从源码仓库直接使用（未安装时）：

```python
import sys; sys.path.insert(0, "/path/to/emotion_classifier")
from src import EmotionClassifier
```

### 常用构造参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `model_dir` | `~/.emotion_classifier/models` | 模型目录；也可用环境变量 `EMOTION_MODEL_DIR` |
| `backend` | `"auto"` | `auto` / `onnx` / `pytorch` |
| `auto_download` | `False` | 缺失模型时是否自动下载 |
| `device` | 自动探测 | `cpu` / `cuda`；不传则查 ONNXRuntime providers，不加载 torch |
| `tokenizer_dir` | `<model_dir>/tokenizer` | 本地 tokenizer 目录 |
| `encoder_path` | `None` | 自定义编码器（ONNX 路径或 SentenceTransformer 名称） |
| `encoder_cache_size` | `0` | 文本向量缓存条数；**交互 / 重复语料场景建议设 1024** |

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

> 上表序号仅作阅读顺序。实际标签 ID 由 `LabelEncoder` 按**字典序**生成（`label_map.json` 中依次为 厌恶、哭泣、害怕、害羞、平静、心动、惊讶、慌张、担心、无奈、生气、疑惑、紧张、自信、认真、调皮、难为情、高兴）。**以训练产出的 `label_map.json` 为准**，不要硬编码 ID。

---

## 3. 命令行用法

```bash
# 单条文本
emotion-classify "脸颊泛红，偷偷瞄了你一眼"

# 指定 top-k
emotion-classify "气鼓鼓地跺着脚瞪着你" --top-k 5

# 批量文本
emotion-classify --batch "text1" "text2" "text3"

# 从文件读取（每行一条）
emotion-classify --file input.txt

# JSON 输出（方便管道处理）
emotion-classify "今天的天气真好" --json

# 交互模式（重复输入场景建议加 --cache 1024）
emotion-classify --interactive --cache 1024

# 指定后端 / 模型目录 / 离线
emotion-classify "test" --backend onnx
emotion-classify "test" --model-dir /path/to/models
emotion-classify "test" --no-download

# 运维
emotion-classify --labels        # 列出所有标签
emotion-classify --info          # 模型状态（后端、设备、标签数、文件完整性）
emotion-classify --list-models   # 列出所有可用分类头/编码器及当前选中的（★）
emotion-classify --update        # 强制重新下载模型
emotion-classify --verbose       # 详细推理信息
```

源码运行时用 `python emotion-classify.py ...` 代替 `emotion-classify ...`。

### CLI 输出示例

```bash
$ emotion-classify "底下头，脸颊泛红，偷偷瞄了你一眼" --top-k 3
「底下头，脸颊泛红，偷偷瞄了你一眼」
  #1 害羞     44.50%  ████████
  #2 疑惑     16.60%  ███
  #3 生气     8.48%  █

$ emotion-classify "私は嬉しくてうなだれた"
「私は嬉しくてうなだれた」
  #1 高兴     69.31%  █████████████
  #2 无奈     6.99%  █
  #3 难为情    3.80% 

$ emotion-classify "今天的天气真好" --json
[
  {
    "text": "今天的天气真好",
    "predictions": [
      {
        "label": "平静",
        "score": 0.4219
      },
      {
        "label": "高兴",
        "score": 0.144
      },
      {
        "label": "自信",
        "score": 0.1249
      }
    ]
  }
]
```

---

## 3. 训练自己的模型

### 数据格式

CSV，至少包含 `text` 与 `label` 两列；可选 `lang` 列（用于语言过滤与分层）。

```csv
text,label,lang
今天心情真好,高兴,zh
i'm so excited i can hardly breathe,高兴,en
怖くて震えが止まらない,害怕,ja
```

- 自动尝试 `utf-8-sig / utf-8 / gbk / latin1` 四种编码；
- 自动清洗零宽字符、全角空格、控制字符、HTML、URL、装饰符号；
- 自动剔除空文本、`(text, label)` 重复、**同文本不同标签的标注冲突样本**。

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--hidden-dim` | `128` | 分类头隐藏层维度 |
| `--dropout` | `0.3` | Dropout 率 |
| `--lr` | `5e-4` | 学习率 |
| `--weight-decay` | `1e-2` | 权重衰减 |
| `--batch-size` | `64` | 批大小 |
| `--epochs` | `120` | 最大轮数 |
| `--early-stop` | `40` | 早停轮数 |
| `--metric` | `macro_f1` | **早停/选模指标**，类别不平衡下请勿用 `accuracy` |
| `--class-weight` | `effective` | `none` / `inverse` / `sqrt` / `effective` |
| `--focal-gamma` | `2.0` | Focal gamma，`0` 退化为加权 CE |
| `--no-focal` | - | 直接关闭 Focal Loss |
| `--mixup-alpha` | `0.2` | MixUp 强度，`0` 关闭 |
| `--label-smoothing` | `0.05` | 标签平滑 |
| `--use-aug` | `True` | 对**训练集**做少数类增强过采样 |
| `--aug-factor` | `2` | 少数类增强倍率 |
| `--lang` | 空 | 只保留指定语言，如 `zh,en`；留空=全部 |
| `--stratify-lang` | `False` | 划分时按 `标签×语言` 分层 |
| `--folds` | `0` | 交叉验证折数，`0`=不启用 |
| `--device` | 自动 | `cuda` / `cpu` |

### 推荐配置

```bash
#设置train模式
python main.py \
  --data data/emotion.csv \
  --hidden-dim 256 --dropout 0.4 --lr 2e-3 \
  --class-weight effective --focal-gamma 2.0 --mixup-alpha 0.2 \
  --label-smoothing 0.05 --metric macro_f1 --epochs 120 --early-stop 40
```

## 项目结构

```text
emotion_classifier/
├── src/                          # 推理包（安装后名为 emotion_classifier）
│   ├── models/
│   │   ├── config.py             # 模型仓库、校验和、标签映射路径解析
│   │   ├── encoder.py            # 批量 ONNX 编码器（核心性能修复）
│   │   └── pytorch.py            # PyTorch 分类头
│   ├── __init__.py
│   ├── cli.py                    # 命令行入口
│   ├── downloader.py             # 断点续传 + SHA256 校验
│   └── predictor.py              
├── train/                        
│   ├── augment.py                # EDA 增强
│   ├── config.py                 # 配置
│   ├── cli.py                   
│   ├── models.py                 # 分类头 / FocalLoss / ONNXEncoder / ChunkedEncoder
│   ├── train.py                  # clean / split / aug / embed / train / eval / onnx / infer
│   └── utils.py                  # set_seed / mixup / class_weights / macro_f1
准
├── models/                       # 模型与 tokenizer（首次运行自动下载）
├── data/                         # 训练数据
├── main.py           # 源码模式 CLI 入口
└── setup.py
```

---

## 模型文件

| 文件 | 说明 |
|------|------|
| `emotion_classify.pt` | PyTorch 模型权重 |
| `emotion_classifier.onnx` | ONNX FP32（**推荐部署用**） |
| `emotion_classifier_fp16.onnx` | ONNX FP16（体积减半；CPU 上会回退为 FP32 计算，准确度基本无损） |
| `emotion_classifier_int8.onnx` | ONNX INT8 动态量化（分类头极小，收益主要是体积；导出后会与 FP32 做数值偏差自检） |
| `model_quint8_avx2.onnx` | 量化编码器（来自 HuggingFace） |
| `label_map.json` | 标签映射文件 |
| `embeddings.npz` | 预计算嵌入缓存（可选） |

- 分类头优先加载顺序：`emotion_classifier.onnx` → `*_fp16.onnx` → `*_int8/quant.onnx` → 其他。
---

## 已知限制

1. **多语言准确率不均衡**：英文/日文整体低于中文（原 README 亦提到「多语言情况预测准确不到 70%」）。修复了分块与归一化后有所改善，但根因是训练数据的语言分布，建议：
   - 用 `--lang zh,en` 分别训练专用模型；
   - 或保证各语言样本量均衡，并用 `--stratify-lang` 分层划分。
2. **短文本歧义**：如 `"I happily lowered my head"` 被判为「害羞」而非「高兴」——句向量对副词位置不敏感，属模型容量限制。
3. **INT8 量化**：分类头本身极小，量化主要收益是体积；导出后会自动与 FP32 做数值偏差自检并打印最大偏差。
4. **FP16 在 CPU 上的行为**：ONNXRuntime CPU 会把 FP16 回退为 FP32 计算，因此**不会加速**，仅在支持 FP16 的后端上有收益。

---

## 4. 集成示例

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

tools = [EmotionAnalysisTool()]
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

> 服务化场景建议开启 `encoder_cache_size`，热点文本可直接命中缓存。

---

## License

MIT

## 致谢

- [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — 多语言句向量编码器
