import onnx
from onnxsim import simplify

model = onnx.load("models/emotion_classifier.onnx")
model_simp, check = simplify(model)

from onnxconverter_common import float16

model_fp16 = float16.convert_float_to_float16(model_simp)
onnx.save(model_fp16, "models/emotion_classifier_fp16.onnx")