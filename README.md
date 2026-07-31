# Emotion Classifier

多语言情感分类模型包，支持 **19 种情感标签**，覆盖 **中文、英文、日文**。

## 安装

> 请注意本模型是基于`paraphrase-multilingual-MiniLM-L12-v2`模型进行编码的，在使用本模型前请先安装该模型,推荐模型`model_quint8_avx2.onnx`轻量。


```bash
pip install emotion-classifier
```

或从源码安装：

```bash
git clone <repo-url>
cd emotion_classifier_pkg
pip install .
```

## 支持的 19 种情感

| 序号 | 标签 | 说明 |
|------|------|------|
| 1 | 兴奋 | 极度高兴、激动 |
| 2 | 厌恶 | 反感、恶心 |
| 3 | 哭泣 | 悲伤哭泣 |
| 4 | 害怕 | 恐惧、惊慌 |
| 5 | 害羞 | 羞涩、脸红 |
| 6 | 平静 | 冷静、安宁 |
| 7 | 心动 | 浪漫、喜欢 |
| 8 | 惊讶 | 吃惊、意外 |
| 9 | 慌张 | 慌乱、手忙脚乱 |
| 10 | 担心 | 焦虑、不安 |
| 11 | 无奈 | 无力、放弃 |
| 12 | 生气 | 愤怒、恼火 |
| 13 | 疑惑 | 困惑、不解 |
| 14 | 紧张 | 局促、不安 |
| 15 | 自信 | 确信、从容 |
| 16 | 认真 | 专注、严肃 |
| 17 | 调皮 | 顽皮、戏弄 |
| 18 | 难为情 | 尴尬、羞耻 |
| 19 | 高兴 | 开心、喜悦 |

## Python API 用法

```python
from emotion_classifier import EmotionClassifier

# 初始化（自动检测 ONNX / PyTorch 后端）
clf = EmotionClassifier()

# 单条预测
result = clf.predict("脸颊泛红，偷偷瞄了你一眼", top_k=3)
print(result)
# [[('害羞', 0.80), ('生气', 0.12), ('高兴', 0.03)]]

# 批量预测
texts = [
    "i'm so excited i can hardly breathe",
    "怖くて震えが止まらない",
    "气鼓鼓地跺着脚瞪着你"
]
results = clf.predict(texts, top_k=2)
for text, preds in zip(texts, results):
    print(f"{text}: {preds}")

# 仅获取最可能的标签
label = clf.predict_label("今天的天气真好")
print(label)  # [('高兴', 0.65)]

# 查看所有标签
print(clf.get_labels())
```

## 命令行用法

本项目不提供命令行，需要请自行打包为可执行文件，相关用法如下：

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

# 交互模式
emotion-classify --interactive

# 指定后端
emotion-classify "test text" --backend onnx
emotion-classify "test text" --backend pytorch

# 指定模型目录
emotion-classify "test" --model-dir /path/to/models

# 列出所有标签
emotion-classify --labels

# 详细模式
emotion-classify "test" --verbose
```

### CLI 输出示例

```bash
$ emotion-classify "底下头，脸颊泛红，偷偷瞄了你一眼" --top-k 3
「底下头，脸颊泛红，偷偷瞄了你一眼」
  #1 害羞    74.57%  ██████████████
  #2 生气    8.11%  █
  #3 无奈    4.80%  

$ emotion-classify "底下头，脸颊泛红，偷偷瞄了你一眼" --json
[
  {
    "text": "底下头，脸颊泛红，偷偷瞄了你一眼",
    "predictions": [
      {
        "label": "害羞",
        "score": 0.7457
      },
      {
        "label": "生气",
        "score": 0.0811
      },
      {
        "label": "无奈",
        "score": 0.048
      }
    ]
  }
]
```

# 训练模型
训练模型需要使用 PyTorch 框架。以下是一个简单的训练脚本示例：
```python
stage_train(
  emb: dict | None = None, # 嵌入字典，用于存储预训练的嵌入向量
  num_labels: int | None = None, # 情感标签数量
)
returns:
  - model: EmotionClassifierNet
```

## 集成到 Agent 系统

### LangChain Tool 示例

```python
from langchain.tools import BaseTool
from emotion_classifier import EmotionClassifier

class EmotionAnalysisTool(BaseTool):
    name = "emotion_analysis"
    description = "分析文本的情感，返回 top-3 情感标签及置信度"

    def __init__(self):
        super().__init__()
        self.clf = EmotionClassifier()

    def _run(self, query: str) -> str:
        predictions = self.clf.predict(query, top_k=3)
        return str(predictions[0])

# 注册到 Agent
tools = [EmotionAnalysisTool()]
```

### 简单 HTTP 服务（FastAPI）

```python
from fastapi import FastAPI
from pydantic import BaseModel
from emotion_classifier import EmotionClassifier

app = FastAPI()
clf = EmotionClassifier()

class Request(BaseModel):
    text: str
    top_k: int = 3

@app.post("/predict")
def predict(req: Request):
    return {"results": clf.predict(req.text, req.top_k)}

# uvicorn main:app --reload
```

## 模型文件

| 文件 | 说明 |
|------|------|
| `emotion_classifier.pt` | PyTorch 模型权重 |
| `emotion_classifier.onnx` | ONNX 模型（推荐部署用） |
| `label_map.json` | 标签映射文件 |
| `embeddings.npz` | 预计算嵌入缓存（可选） |

## 依赖

- Python >= 3.8
- torch >= 1.12
- sentence-transformers >= 2.2
- onnxruntime >= 1.14
- numpy >= 1.21
- torch>=1.12
- scikit-learn>=1.0
- httpx>=0.24
- tqdm>=4.62

## License

MIT
