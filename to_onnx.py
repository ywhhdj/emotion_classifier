# from sentence_transformers import SentenceTransformer


# st = SentenceTransformer("paraphrase_multilingual_MiniLM_L12_v2")
# st.save_pretrained("models/minilm_onnx",)

# 2. 对导出的 ONNX 做动态量化
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    model_input="models/emotion_classifier.onnx",
    model_output="models/emotion_classifier_int8.onnx",
    weight_type=QuantType.QInt8
)