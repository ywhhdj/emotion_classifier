"""
验证词表裁剪的正确性（无需 torch / transformers / 网络）。

用**合成的 XLM-R 结构**（250,002 词、6 个特殊 token、SentencePieceBPE）
跑真实的 train/vocab_trim.py 代码，断言：

  A. 保留集规划
     1. 特殊 token 全部保留，且 ID 不变（0..5 → 0..5）
     2. 语料命中的 token 全部保留
     3. --keep-chars 能补齐所有单字符 token（含 ▁ 前缀）
     4. --min-size 能把词表补到下限
     5. 新旧 ID 映射双射且连续

  B. tokenizer.json 重建
     6. 新 vocab 长度 == 新词表大小
     7. 新 vocab[i] 的内容 == 旧 vocab[new_to_old[i]]
     8. added_tokens 的 id 被重映射且不越界
     9. 若漏掉特殊 token → 明确报错（而不是产出坏文件）

  C. 权重裁剪
    10. 嵌入矩阵行数 == 新词表大小（对齐，否则索引越界）
    11. 新嵌入第 i 行 == 旧嵌入第 new_to_old[i] 行（内容逐字节一致）
    12. find_vocab_tensors 能同时识别嵌入矩阵与 LM head，且不误伤普通层

  D. 端到端一致性 + 守卫
    13. 旧 ID 序列 → 新 ID → 查新词表，保留的 token 完全一致
    14. UNK 率计算正确
    15. 编码器守卫能拦下 [B,S,250002] 这类「词表维度当隐藏维度」的污染
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

import train.vocab_trim as VT  # noqa: E402

CHECKS = []
HIDDEN = 16          # 测试用小 hidden，省内存（真实是 768）
VOCAB = 250_002      # 与真实 XLM-R 一致
SPECIALS = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<mask>"]
N_SPECIAL = len(SPECIALS)


def ok(cond, label):
    CHECKS.append(bool(cond))
    print(f"  {'✅' if cond else '❌'} {label}")
    return bool(cond)


# ── 构造合成词表 ──
def build_fake_vocab():
    """0..5 特殊 token；其余混合：单字符（含 ▁ 前缀）与多字符片段。"""
    toks = list(SPECIALS)
    single = ["中", "日", "あ", "a", "é", "。", "▁中", "▁a", "▁日"]
    multi = []
    n_single = 5000
    for i in range(n_single):
        toks.append(f"{single[i % len(single)]}{i // len(single)}"
                    if i >= len(single) else single[i])
    # 保证前 n_single 个里混入真正的单字符；其余多字符
    while len(toks) < VOCAB:
        toks.append(f"▁frag_{len(toks)}")
    return toks


VOCAB_TOKENS = build_fake_vocab()
SINGLE_IDS = {i for i, t in enumerate(VOCAB_TOKENS)
              if len(VT._strip_metaspace(t)) == 1}


def fake_tokenizer_json():
    return {
        "version": "1.0",
        "added_tokens": [{"id": i, "special": True, "content": t,
                          "normalized": True, "lstrip": False,
                          "rstrip": False, "single_word": False}
                         for i, t in enumerate(SPECIALS)],
        "normalizer": {"type": "Precompiled", "precompiled_charsmap": None},
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁",
                          "prepend_scheme": "always", "split": True},
        "post_processor": {"type": "RobertaProcessing",
                           "sep": ["</s>", 2], "cls": ["<s>", 0],
                           "add_prefix_space": True, "trim_offsets": True},
        "decoder": {"type": "Metaspace", "replacement": "▁",
                    "prepend_scheme": "always", "split": True},
        "model": {
           "type": "SentencePieceBPE",
           "dropout": None, 
           "unk_token": None,
            "fuse_unk": True,
            "byte_fallback": False,
            "vocab": [[t, 0.0] for t in VOCAB_TOKENS],
            "merges": None
         },
    }


def main():
    print("=" * 74)
    print(f"词表裁剪验证（合成 XLM-R：{VOCAB:,} 词 / {N_SPECIAL} 特殊 token）")
    print("=" * 74)

    rng = np.random.default_rng(42)
    # 语料命中：随机 1200 个内容 token（避开特殊位）
    seen = set(rng.choice(np.arange(N_SPECIAL, VOCAB), size=1200,
                          replace=False).tolist())
    seen |= {0, 2}   # <s> </s> 必然出现

    # ══ A. 规划 ══
    print("\n[A] 保留集规划")
    plan = VT.build_plan(VOCAB_TOKENS, seen)
    ok(all(i in plan.old_to_new for i in range(N_SPECIAL)),
       f"{N_SPECIAL} 个特殊 token 全部保留")
    ok(all(plan.old_to_new[i] == i for i in range(N_SPECIAL)),
       "特殊 token 的 ID 保持不变（0..5 → 0..5）")
    ok(all(i in plan.old_to_new for i in seen), "语料命中的 token 全部保留")
    ok(plan.new_size == len(set(plan.kept_old_ids)), "保留集无重复")
    ok(plan.kept_old_ids == sorted(plan.kept_old_ids), "保留集升序 → 新 ID 连续")

    # 双射
    ok(all(plan.new_to_old[plan.old_to_new[o]] == o for o in plan.kept_old_ids[:2000]),
       "old→new→old 往返一致")
    ok(len(plan.new_to_old) == plan.new_size, "new→old 映射完整")

    # keep_chars
    plan_chars = VT.build_plan(VOCAB_TOKENS, seen, keep_single_chars=True)
    ok(SINGLE_IDS <= set(plan_chars.old_to_new),
       f"--keep-chars 补齐全部 {len(SINGLE_IDS):,} 个单字符 token")
    ok(plan_chars.new_size > plan.new_size, "--keep-chars 后词表更大")

    # min_size
    plan_min = VT.build_plan(VOCAB_TOKENS, seen, min_size=20000)
    ok(plan_min.new_size >= 20000, f"--min-size 生效（{plan_min.new_size:,} ≥ 20000）")
    ok(all(plan.old_to_new[i] == i for i in range(N_SPECIAL)),
       "--min-size 补齐后特殊 token ID 仍未变")

    # ══ B. tokenizer.json 重建 ══
    print("\n[B] tokenizer.json 重建")
    tj = fake_tokenizer_json()
    new_tj = VT.apply_to_tokenizer_json(tj, plan)
    new_vocab = new_tj["model"]["vocab"]
    ok(len(new_vocab) == plan.new_size,
       f"新 vocab 长度 {len(new_vocab):,} == 新词表大小")
    ok(all(new_vocab[i][0] == VOCAB_TOKENS[plan.new_to_old[i]]
           for i in range(0, plan.new_size, 97)),
       "新 vocab[i] 内容 == 旧 vocab[new_to_old[i]]")
    ok(len(tj["model"]["vocab"]) == VOCAB, "原 tokenizer.json 未被修改（深拷贝）")

    at_ids = [at["id"] for at in new_tj["added_tokens"]]
    ok(max(at_ids) < plan.new_size, f"added_tokens 的 id 不越界（max={max(at_ids)}）")
    ok(at_ids == list(range(N_SPECIAL)), "特殊 token 仍占 0..5")
    ok(all(new_tj["model"]["vocab"][at["id"]][0] == at["content"]
           for at in new_tj["added_tokens"]),
       "added_tokens 的 id 与其在 vocab 中的字符串严格对齐")

    # 漏掉特殊 token → 报错而不是产出坏文件
    bad_plan = VT.build_plan(VOCAB_TOKENS, seen,
                             special_ids=[0, 1])  # 故意漏 <unk> 等
    try:
        VT.apply_to_tokenizer_json(tj, bad_plan)
        ok(False, "漏掉特殊 token 时应报错")
    except ValueError as e:
        ok("不在保留集" in str(e), f"漏掉特殊 token 时明确报错: {str(e)[:40]}...")

    # ══ C. 权重裁剪 ══
    print("\n[C] 权重裁剪")
    emb = rng.random((VOCAB, HIDDEN), dtype=np.float32)   # [250002, 16]
    new_emb = VT.trim_array(emb, plan)
    ok(new_emb.shape == (plan.new_size, HIDDEN),
       f"嵌入矩阵 {emb.shape} → {new_emb.shape}（行数 == 新词表大小）")
    sample = rng.choice(np.arange(plan.new_size), size=300, replace=False)
    ok(np.array_equal(new_emb[sample], emb[[plan.new_to_old[i] for i in sample]]),
       "新嵌入第 i 行 == 旧嵌入第 new_to_old[i] 行（逐元素一致）")
    ok(not np.shares_memory(new_emb, emb), "裁剪结果是拷贝，不与原矩阵共享内存")

    # 1-D 的 LM head bias
    bias = rng.random(VOCAB, dtype=np.float32)
    new_bias = VT.trim_array(bias, plan)
    ok(new_bias.shape == (plan.new_size,), f"1-D 的 lm_head.bias 也能裁剪")
    ok(np.array_equal(new_bias[sample], bias[[plan.new_to_old[i] for i in sample]]),
       "bias 行索引一致")

    # find_vocab_tensors 识别能力
    state = {
        "embeddings.word_embeddings.weight": emb,
        "embeddings.position_embeddings.weight": np.zeros((514, HIDDEN), np.float32),
        "embeddings.token_type_embeddings.weight": np.zeros((1, HIDDEN), np.float32),
        "encoder.layer.0.attention.self.query.weight": np.zeros((HIDDEN, HIDDEN), np.float32),
        "lm_head.decoder.weight": emb,
        "lm_head.decoder.bias": bias,
    }
    found = VT.find_vocab_tensors(state, VOCAB)
    ok(set(found) == {"embeddings.word_embeddings.weight",
                      "lm_head.decoder.weight", "lm_head.decoder.bias"},
       f"识别出全部词表相关张量（{len(found)} 个），未误伤普通层")
    ok("encoder.layer.0.attention.self.query.weight" not in found,
       "普通层未被误判为词表张量")

    new_state, targets = VT.trim_state_dict(state, plan, verbose=False)
    ok(all(np.asarray(new_state[n]).shape[0] == plan.new_size for n in targets),
       "trim_state_dict 后所有相关张量首维对齐")
    ok(np.asarray(state["embeddings.position_embeddings.weight"]).shape == (514, HIDDEN),
       "position_embeddings 未被改动（与词表无关）")

    # ══ D. 端到端 + 守卫 ══
    print("\n[D] 端到端一致性与守卫")
    test_ids = list(rng.choice(np.arange(N_SPECIAL, VOCAB), 500, replace=False))
    mapped = [plan.old_to_new[int(i)] for i in test_ids if int(i) in plan.old_to_new]
    ok(all(new_tj["model"]["vocab"][m][0] == VOCAB_TOKENS[int(i)]
           for i, m in zip([t for t in test_ids if int(t) in plan.old_to_new], mapped)),
       "旧 ID → 新 ID → 查新词表，token 字符串完全一致")

    # UNK 率
    in_ids = [i for i in test_ids if int(i) in plan.old_to_new]
    mixed = in_ids + [-1, -2, -3, -4]           # 4 个必然不在保留集
    r = VT.unk_rate(mixed, plan)
    ok(abs(r - 4 / len(mixed)) < 1e-9, f"UNK 率计算正确（{r * 100:.2f}%）")
    ok(VT.unk_rate([], plan) == 0.0, "空输入的 UNK 率为 0（不除零）")

    # 守卫：拦下 [B,S,250002]
    spec = importlib.util.spec_from_file_location(
        "enc_guard", ROOT / "src" / "models" / "encoder.py")
    enc_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc_mod)
    guard = enc_mod.ONNXEncoder._reject_bad_output

    ok(guard(np.zeros((2, 768), np.float32), 768, "t").shape == (2, 768),
       "守卫放行正常的 2-D [batch, hidden]")
    ok(guard(np.zeros((2, 8, 768), np.float32), 768, "t").shape == (2, 8, 768),
       "守卫放行正常的 3-D [batch, seq, hidden]")

    try:
        guard(np.zeros((2, 8, 250002), np.float32), 768, "测试")
        ok(False, "守卫应拦下 [B,S,250002]")
    except RuntimeError as e:
        msg = str(e)
        ok("250002" in msg and "250002 != 期望 768" in msg,
           "守卫拦下 [B,S,250002] 并指明实际/期望维度")
        ok("LM head" in msg, "守卫给出「带 LM head 的原始模型」的针对性提示")
        ok("--export-encoder" in msg, "守卫给出正确的修复命令")

    try:
        guard(np.zeros((2, 3, 4, 5), np.float32), 768, "测试")
        ok(False, "守卫应拦下 4-D 输出")
    except RuntimeError as e:
        ok("维度数异常" in str(e), "守卫拦下维度数异常的输出")

    # 体积收益
    saved = (plan.old_size - plan.new_size) * 768 * 4 / 1e6
    print(f"\n  ── 参考 ──")
    print(f"  本例裁剪: {plan.old_size:,} → {plan.new_size:,} "
          f"(保留 {plan.compression * 100:.2f}%)")
    print(f"  嵌入参数: {(plan.old_size - plan.new_size) * 768:,} 个")
    print(f"  体积节省: ≈ {saved:.0f} MB (fp32)")

    passed = sum(CHECKS)
    total = len(CHECKS)
    print("\n" + "=" * 74)
    print(f"===== 词表裁剪验证结果：{passed}/{total} 通过 =====")
    print("=" * 74)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
