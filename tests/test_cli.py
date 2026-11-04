from emotion_classifier import EmotionClassifier
clf = EmotionClassifier(
    model_dir="../models",
    backend="rust",
    tokenizer_dir="../models/xlmr-zh-en-ja",
)
result = clf.predict("我今天很开心！")
print(result)