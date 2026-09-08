import random

EMOTION_EMOJI = {
    "高兴": ["😊", "😄", "🥰", "😁"],
    "悲伤": ["😭", "😢", "🥲"],
    "愤怒": ["😡", "🤬"],
    "害羞": ["😳", "☺️"],
    "害怕": ["😱", "😨"],
    "惊讶": ["😲", "😮"],
}
PREFIX = ["真的", "非常", "有点", "超级", "今天", "突然"]
SUFFIX = ["啊", "呀", "！！", "..."]


class EDAAugmentor:

    CJK = "\u4e00\u9fff"

    def _is_cjk(self, text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def random_delete(self, text, p=0.1):
        chars = list(text)
        if len(chars) <= 3:
            return text
        out = [c for c in chars if random.random() > p]
        return "".join(out) if out else text

    def random_swap(self, text, n=2):
        if not self._is_cjk(text):  # 英文按 token 交换，避免破坏单词
            units = text.split()
            if len(units) < 4:
                return text
            for _ in range(n):
                i, j = random.randrange(len(units)), random.randrange(len(units))
                units[i], units[j] = units[j], units[i]
            return " ".join(units)
        chars = list(text)
        if len(chars) < 4:
            return text
        for _ in range(n):
            i, j = random.randrange(len(chars)), random.randrange(len(chars))
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

    def emoji(self, text, label=None, p=0.25):
        if random.random() > p:
            return text
        pool = EMOTION_EMOJI.get(label) if label else None
        if not pool:
            pool = [e for v in EMOTION_EMOJI.values() for e in v]
        return f"{text}{random.choice(pool)}"

    def affix(self, text, p=0.2):
        if random.random() > p:
            return text
        if random.random() < 0.5:
            return f"{random.choice(PREFIX)}{text}"
        return f"{text}{random.choice(SUFFIX)}"

    def eda_combine(
        self,
        text: str,
        p_delete=0.1, p_swap=0.1, p_insert=0.1,
        p_punct=0.2, p_emoji=0.15, p_affix=0.15, label=None
    ) -> str:
        """组合增强：删除 / 交换 / 插入 / 标点 / emoji / 前后缀，可叠加。"""
        aug = text
        if random.random() < p_delete:
            aug = self.random_delete(aug, p_delete)
        if random.random() < p_swap:
            aug = self.random_swap(aug)
        if random.random() < p_insert:
            aug = self.random_insert(aug, p_insert)
        if random.random() < p_punct:
            aug = self.punctuation(aug)
        if random.random() < p_emoji:
            aug = self.emoji(aug, label)
        if random.random() < p_affix:
            aug = self.affix(aug)
        return aug if aug != text else text