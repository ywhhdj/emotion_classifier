import random

EMOTION_EMOJI = {
    "happy": ["😊", "😄", "🥰", "😁"],
    "sad": ["😭", "😢", "🥲"],
    "angry": ["😡", "🤬"]
}

PREFIX = ["真的", "非常", "有点", "超级", "今天", "突然"]
SUFFIX = ["啊", "呀", "！！", "..."]


class EDAAugmentor:
    def random_delete(self, text, p=0.1):
        chars = list(text)
        if len(chars) <= 3:
            return text
        out = [c for c in chars if random.random() > p]
        return "".join(out) if out else text

    def random_swap(self, text, n=2):
        chars = list(text)
        if len(chars) < 4:
            return text
        for _ in range(n):
            i = random.randint(0, len(chars) - 1)
            j = random.randint(0, len(chars) - 1)
            chars[i], chars[j] = chars[j], chars[i]
        return "".join(chars)

    def punctuation(self, text):
        mapping = {"!": "！！", "?": "？？", "。": "！"}
        for k, v in mapping.items():
            if random.random() < 0.5:
                text = text.replace(k, v)
        return text

    def random_insert(self, text, p=0.1):
        chars = list(text)
        if len(chars) <= 3:
            return text
        num_insert = max(1, int(p * len(chars)))
        for _ in range(num_insert):
            pos = random.randint(0, len(chars))
            char = random.choice(chars)
            chars.insert(pos, char)
        return "".join(chars)

    def eda_combine(self, text, p_delete=0.1, p_swap=0.1, p_insert=0.1, p_punct=0.2):
        """
        组合 EDA 增强：删除、交换、插入、标点替换
        每种操作独立按概率触发，可叠加
        """
        aug = text
        if random.random() < p_delete:
            aug = self.random_delete(aug, p_delete)
        if random.random() < p_swap:
            aug = self.random_swap(aug)
        if random.random() < p_insert:
            aug = self.random_insert(aug, p_insert)
        if random.random() < p_punct:
            aug = self.punctuation(aug)
        return aug if aug != text else text