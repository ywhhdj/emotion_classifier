# Emotion Classifier

多语言轻量情感分类模型包，支持 **18 种情感标签**，覆盖 **中文 / 英文 / 日文**。

- **推理内核是 Rust**：分词 / 动态 padding / 批调度 / 前向 / 后处理都在 `rust/` 完成，通过 PyO3 暴露；`src/` 是 Python 对外 API（胶水层）。
- **三档 ONNX 整图**，文件名固定，统一从 GitHub Release `v0.2.0` 下载：
  `emotion_classifier.onnx` / `emotion_classifier_fp16.onnx` / `emotion_classifier_int8.onnx`。
- **进程内单一实例**：无论创建多少个 `EmotionClassifier`，模型只初始化一次；也支持把**已加载的模型**交给库、由程序管理生命周期。

---

## 1. 安装

```bash
# 含 Rust 内核（需要 Rust 工具链；默认启用 backend-ort）
pip install .

# 纯 Python 回退（无 Rust 工具链）
EMOTION_BACKEND=onnx pip install .

# 训练侧额外依赖
pip install ".[train]"
```

---

## 2. 快速开始

```python
from src import EmotionClassifier            # 安装后：from emotion_classifier import EmotionClassifier

clf = EmotionClassifier(auto_download=True)  # 缺文件时从 Release 下载
print(clf.predict("脸颊泛红，偷偷瞄了你一眼", top_k=3))
# [[('害羞', 0.93), ('疑惑', 0.04), ('难为情', 0.02)]]

print(clf.predict(["i'm so excited", "怖くて震えが止まらない"], top_k=2))
print(clf.predict_label("今天的天气真好"))
print(clf.predict_proba(["a", "b"]).shape)   # (2, 18)
print(clf.predict("今日はとても嬉しいです", lang="ja"))
print(clf.get_labels())
```

### 指定精度档位

```python
clf = EmotionClassifier(auto_download=True, precision="int8")   # fp32（默认）/ fp16 / int8
clf.model_info()["precision"]                                   # -> "int8"
```

### 传入已加载模型（模型交给程序管理）

```python
import onnxruntime as ort
from src import EmotionClassifier, LoadedModel, release_model

sess = ort.InferenceSession(
  "models/deploy/emotion_classifier_int8.onnx",
    providers=["CPUExecutionProvider"]
)

# 方式 A：完整 bundle（自带 tokenizer / label_map，可完全离线）
clf = EmotionClassifier(model=LoadedModel(
  session=sess,
  tokenizer="models/deploy/tokenizer",
  label_map=None,          # 不传则从 model_dir 读 label_map.json
  precision="int8",
  backend="onnx",
  owns=False,              # 生命周期归调用方，库不会擅自 close
))

# 方式 B：直接传 session
clf = EmotionClassifier(
  model=sess,
  precision="int8",
  tokenizer="models/tokenizer"
)

# 方式 C：传 ONNX 字节或本地路径（库创建 session，release_model() 可回收）
clf = EmotionClassifier(
  model=open("emotion_classifier_fp16.onnx", "rb").read(),
  precision="fp16"
)
clf = EmotionClassifier(model_path="/path/to/emotion_classifier.onnx")

# Rust 内核句柄同样可以注入
from src import _core  # 仅在已编译内核的 wheel 中可用
engine = _core.EmotionEngine(model_dir="models/deploy", precision="int8")
clf = EmotionClassifier(model=engine, precision="int8")
```

`model` 支持：`LoadedModel` / `onnxruntime.InferenceSession` / `_core.EmotionEngine` / ONNX 字节 / `.onnx` 路径。

> 注入 ONNX 字节 / 路径时，库仍需取得分词器（部署目录或 `tokenizer=` 入参）；要完全离线且自带分词器，优先注入 `LoadedModel`。

### 单一实例规则（重要）

| 场景 | 行为 |
|---|---|
| 两个 `EmotionClassifier()` 用相同配置 | 复用同一份模型，**不重复初始化** |
| 配置不同（目录 / 档位 / 线程 / `max_seq_len` 等） | 抛 `BackendError`，不会静默再加载一份 |
| 已注入外部模型，又想从 Release 加载 | 先 `release_model()` 或传 `replace=True` |
| `clf.close()` / `with` 退出 | 只解除本对象引用，**不销毁共享实例** |
| `EmotionClassifier.release_shared_model()` / `release_model()` | 显式释放唯一实例（含 Rust 全局引擎） |
| `owns=False` 的注入资源 | 库绝不主动 `close()`；由程序决定何时释放 |

```python
from src import EmotionClassifier, release_model

a = EmotionClassifier(auto_download=True)
b = EmotionClassifier(auto_download=True)   # 复用 a 的引擎
assert a.model_info()["singleton"] is True

release_model()                              # 换档 / 热更新的出口
c = EmotionClassifier(auto_download=True, precision="fp16")
```

### 构造参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `model_dir` | `~/.emotion_classifier/models` | 部署目录；环境变量 `EMOTION_MODEL_DIR` |
| `backend` | `"auto"` | `auto`（Rust→ONNX→PyTorch）/ `rust` / `onnx` / `pytorch` |
| `precision` | `"auto"` | `auto` / `fp32` / `fp16` / `int8`，对应三档 ONNX |
| `model` / `model_path` | `None` | 注入已加载模型（见上） |
| `owns_model` | `False` | 注入资源是否由库负责释放 |
| `tokenizer` / `label_map` / `calib` | `None` | 注入配套资源，离线场景可完全不落盘 |
| `auto_download` | `False` | 缺失资产时从 Release 下载 |
| `repo_url` | Release `v0.2.0` | 自定义 Release / 镜像地址 |
| `use_lang_token` | `None` | 显式 `True/False` > `deploy_manifest.json` > 训练默认 `False` |
| `max_seq_len` | `None` | 同上，默认 `64`（与 `train/config.py` 一致） |
| `num_threads` / `max_batch` | `4` / `32` | Rust 内核线程数与批上限 |
| `device` | 自动 | `cpu` / `cuda` |
| `tokenizer_dir` / `encoder_path` / `encoder_cache_size` | — | 旧两段式产物兼容项 |
| `replace` | `False` | 允许先释放旧单例再登记新模型 |

对外方法保持不变：`predict` / `predict_batch` / `predict_proba` / `predict_label` / `get_labels` / `model_info` / `update_models` / `close` / 上下文管理器 / `__call__`。

### 18 种情感标签

高兴、厌恶、哭泣、害怕、害羞、平静、心动、惊讶、慌张、担心、无奈、生气、疑惑、紧张、自信、认真、调皮、难为情。
标签 ID 由 `LabelEncoder` 按字典序固化在 `label_map.json`，**以部署目录中的文件为准**。

---

## 3. 命令行

```bash
emotion-classify "脸颊泛红，偷偷瞄了你一眼"
emotion-classify "test" --precision int8        # 指定档位
emotion-classify "test" --model ./models/deploy/emotion_classifier_fp16.onnx
emotion-classify --batch "text1" "text2" --top-k 5 --json
emotion-classify --file input.txt --backend onnx
emotion-classify --interactive --cache 1024
emotion-classify --labels / --info / --list-models / --update
emotion-classify --no-download                 # 纯离线
emotion-classify --max-seq-len 64 --lang-token
```

#### CLI 输出示例
```bash
$ emotion-classify "底下头，脸颊泛红，偷偷瞄了你一眼" --top-k 3
「底下头，脸颊泛红，偷偷瞄了你一眼」
  #1 害羞     96.34%  ███████████████████
  #2 厌恶     0.74%  
  #3 担心     0.72%  

$ emotion-classify "私は嬉しくてうなだれた"
「私は嬉しくてうなだれた」
  #1 高兴     97.65%  ███████████████████
  #2 自信     1.49%  
  #3 心动     0.23%

#明显的不足
$ emotion-classify "今天的天气真好" --json
[
  {
    "text": "今天的天气真好",
    "predictions": [
      {
        "label": "生气",
        "score": 0.2855
      },
      {
        "label": "高兴",
        "score": 0.1672
      },
      {
        "label": "自信",
        "score": 0.1033
      }
    ]
  }
]
```
---

## 4. Release 资产与部署目录

`auto_download=True` 时下载的最小集合：

| 资产 | 说明 | 必需 |
|---|---|---|
| `emotion_classifier.onnx` | FP32 整图（默认档） | ✅（三档至少有其一） |
| `emotion_classifier_fp16.onnx` | FP16 整图 | 可选 |
| `emotion_classifier_int8.onnx` | INT8 动态量化 | 可选 |
| `label_map.json` | 18 标签映射 | ✅ |
| `tokenizer.zip` | HF tokenizer，解压到 `<model_dir>/tokenizer/` | ✅（本地无 tokenizer 时） |
| `deploy_manifest.json` | 训练参数 / 图签名 | 可选（缺失回退训练默认） |
| `checksums.json` | SHA256，下载后逐项校验 | 可选 |
| `calib.json` | 温度 / per-class 阈值 | 可选 |


部署目录（下载后）：

```text
<model_dir>/
├── emotion_classifier.onnx
├── emotion_classifier_fp16.onnx
├── emotion_classifier_int8.onnx
├── tokenizer/                    # 由 tokenizer.zip
├── label_map.json
├── calib.json
├── deploy_manifest.json
└── checksums.json
```

**推理侧绝不联网拉 hub**：tokenizer 只从部署目录读，缺失就报错并提示下载 Release 资产。

---

## 5. 与训练 `train/` 的对齐点

| 项 | 训练侧 | 推理侧（v0.2.0） |
|---|---|---|
| 基座 | `xlm-roberta-base`（768 维） | 整图 ONNX，输入 `input_ids` / `attention_mask`，输出 `logits` |
| 导出 | `stage_onnx`，opset 17，动态 batch/sequence | Rust `OrtBackend` / Python ORT 直接加载 |
| `MAX_SEQ_LEN` | `train/config.py: 64` | `ModelConfig.MAX_SEQ_LEN = 64`（manifest 可覆盖） |
| `USE_LANG_TOKEN` | 默认 `False` | 默认 `False`（manifest / 显式参数可覆盖） |
| padding | `make_collate` 按 batch 内最长动态 pad | `Tokenizer` 返回实际长度，`Engine::pad_batch` 动态 pad |
| ORT 优化级别 | `_ort_session` 用 `ORT_ENABLE_BASIC`（避开 SkipLayerNormFusion） | Rust `Level1`、Python `ORT_ENABLE_BASIC`；`EMO_ORT_OPT_LEVEL=all` 可覆盖 |
| `position_ids` | HF `create_position_ids_from_input_ids`（pad+1 起算） | `emotion-runtime::create_position_ids`（candle 后端） |
| 校准 | `stage_calibrate` → `calib.json` | Python / Rust 双双读取温度缩放 |

三档回退优先级仅用于 `precision="auto"`（`fp32 → fp16 → int8`）；显式指定档位时严格按该档文件名下载/加载，缺档就提示 `auto_download=True` 补齐，不再静默回退到本地已有的其它档位。

---

## 6. 训练自己的模型

数据为 CSV（`text,label,lang`），当前 `data/emotion.csv` 共 2556 行（zh 1863 / en 356 / ja 337）。

```bash
python main.py --train-mode e2e --use-lang-token False --max-seq-len 64
python main.py --stage onnx            # 产出三档 ONNX + calib.json
python scripts/export/package_deploy.py --onnx-dir model --out models/deploy
```

训练细节（LoRA、回译、跨语言扩充、词表裁剪、剪枝蒸馏、实验矩阵）见 `docs/改造设计_v2.md` 与 `docs/MIGRATION_v1_to_v2.md`。ONNX 是**导出产物**，不是训练输入；训练必须从 HF 权重目录 `from_pretrained`。

---

## 7. 项目结构

```text
src/                         # Python 对外 API
├── predictor.py             # EmotionClassifier（注入 / 单例 / 三档选择）
├── runtime.py               # 进程内唯一实例 + 外部模型登记
├── downloader.py            # Release 资产下载（断点续传 / zip / 校验）
├── models/
│   ├── config.py            # 版本、三档文件名、部署清单、参数默认值
│   ├── loaded.py            # LoadedModel（已加载模型容器）
│   └── encoder.py / pytorch.py
├── languages.py / threshold.py / exceptions.py / legacy.py / cli.py
└── data/label_map.json
emotion_classifier/           # 安装后的对外包名（src 的薄别名）
native/crates/
├── emotion-core             # 类型、三档常量、单例配置指纹
├── emotion-tokenizer        # SentencePiece + 训练侧清洗 + 动态长度
├── emotion-runtime          # 全局单一引擎、ort/candle 后端、批调度
├── emotion-postprocess      # softmax / 温度 / top-k
├── emotion-python           # PyO3 → src._core
└── emotion-cli              # check / bench / run
train/                       # 训练与 ONNX 导出
```

---

1. **en / ja 数据稀疏**：每类 16~36 条，精度瓶颈在数据，解法是回译与跨语言扩充。
2. **FP16 在 CPU 上不加速**：ORT CPU 会回退 FP32 计算，收益主要是体积。
3. **INT8 掉点**：量化后必须重跑三语评测，掉点 > 0.5 pt 就退回 FP16。
4. **candle 后端是可选档**：Release 只发 ONNX；要用纯 Rust 前向需额外提供 `model.safetensors` 并显式 `--backend candle`。
5. **`transformers` 仅在 Python ONNX 回退路径需要**：Rust 内核路径不导入它，也不联网。

---

## 9. 集成示例（长生命周期服务只需一次加载）

```python
from fastapi import FastAPI
from pydantic import BaseModel
from src import EmotionClassifier

app = FastAPI()
clf = EmotionClassifier(auto_download=True, precision="int8")  # 进程级唯一实例

class Req(BaseModel):
  text: str
  top_k: int = 3

@app.post("/predict")
def predict(req: Req):
  return {"results": clf.predict(req.text, req.top_k)[0]}
```

---

## License

MIT 见[LICENSE](LICENSE)

## 致谢

[xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) · [TextPruner](https://github.com/airaria/TextPruner) · [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) · [PEFT](https://github.com/huggingface/peft) · [ort](https://ort.pyke.io/) / [candle](https://github.com/huggingface/candle) / [PyO3](https://pyo3.rs)
