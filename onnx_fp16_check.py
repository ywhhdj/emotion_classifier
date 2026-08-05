import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
from onnxsim import simplify
import argparse

def fp16_convert_and_check(onnx_path, output_path=None, num_samples=100, atol=0.01, rtol=0.05):
    """
    将 FP32 ONNX 模型转为 FP16，并进行精度检测。
    """
    if output_path is None:
        output_path = onnx_path.replace(".onnx", "_fp16.onnx")
    print(f"[check] 加载模型: {onnx_path}")
    model = onnx.load(onnx_path)
    model_simp, check = simplify(model)
    if not check:
        print("[check] 简化失败，继续使用原模型")
        model_simp = model
    else:
        print("[check] 简化成功")
    print("[check] 转换为 FP16...")
    model_fp16 = float16.convert_float_to_float16(model_simp)
    onnx.save(model_fp16, output_path)
    print(f"[check] FP16 模型已保存: {output_path}")
    
    sess_fp32 = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    sess_fp16 = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    
    input_name = sess_fp32.get_inputs()[0].name
    input_shape = sess_fp32.get_inputs()[0].shape
    input_shape = [1 if dim == "batch" else dim for dim in input_shape]
    print(f"[check] 输入形状: {input_shape}")
    
    max_diff = 0.0
    max_rel_diff = 0.0
    mismatches = 0
    
    for i in range(num_samples):
        x = np.random.randn(*input_shape).astype(np.float32) * 0.5
        # FP32 推理
        out_fp32 = sess_fp32.run(None, {input_name: x})[0]
        # FP16 推理
        x_fp16 = x.astype(np.float16)
        out_fp16 = sess_fp16.run(None, {input_name: x_fp16})[0]
        if out_fp16.dtype == np.float16: # type: ignore
            out_fp16 = out_fp16.astype(np.float32) # type: ignore
        # 计算差异
        diff = np.abs(out_fp32 - out_fp16) # type: ignore
        rel_diff = diff / (np.abs(out_fp32) + 1e-8) # type: ignore
        max_diff = max(max_diff, diff.max())
        max_rel_diff = max(max_rel_diff, rel_diff.max())
        if out_fp32.argmax() != out_fp16.argmax(): #type: ignore
            mismatches += 1

    print("\n========== FP16 精度检测报告 ==========")
    print(f"测试样本数: {num_samples}")
    print(f"最大绝对误差: {max_diff:.6f}")
    print(f"最大相对误差: {max_rel_diff:.6f}")
    print(f"Argmax 不一致次数: {mismatches}/{num_samples}")
    if mismatches == 0 and max_rel_diff < rtol:
        print("✅ 检测通过：FP16 模型精度合格，可安全使用。")
    else:
        print("⚠️  检测警告：FP16 模型存在较大偏差，建议检查模型结构或使用 FP32。")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP16 ONNX 转换与精度检测")
    parser.add_argument("--onnx_path", default="models/emotion_classifier.onnx", help="输入 FP32 ONNX 模型路径")
    parser.add_argument("--output", default="models/emotion_classifier_fp16.onnx", help="输出 FP16 ONNX 模型路径")
    parser.add_argument("--samples", type=int, default=100, help="测试样本数")
    args = parser.parse_args()
    
    fp16_convert_and_check(args.onnx_path, args.output, num_samples=args.samples)