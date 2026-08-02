import random
EMOTION_EMOJI={
    "happy":[
        "😊",
        "😄",
        "🥰",
        "😁"
    ],
    "sad":[
        "😭",
        "😢",
        "🥲"
    ],
    "angry":[
        "😡",
        "🤬"
    ]
}

PREFIX=[
    "真的",
    "非常",
    "有点",
    "超级",
    "今天",
    "突然"
]

SUFFIX=[
    "啊",
    "呀",
    "！！",
    "..."
]

class EDAAugmentor:

    def random_delete(self,text,p=0.1):
        chars=list(text)
        if len(chars)<=3:
            return text
        out=[]
        for c in chars:
            if random.random()>p:
                out.append(c)
        return "".join(out) if out else text

    def random_swap(self,text,n=2):
        chars=list(text)
        if len(chars)<4:
            return text
        for _ in range(n):
            i=random.randint(0,len(chars)-1)
            j=random.randint(0,len(chars)-1)
            chars[i],chars[j]=chars[j],chars[i]
        return "".join(chars)

    def punctuation(self,text):
        mapping={
            "!":"！！",
            "?":"？？",
            "。":"！"
        }
        for k,v in mapping.items():
            if random.random()<0.5:
                text=text.replace(k,v)
        return text