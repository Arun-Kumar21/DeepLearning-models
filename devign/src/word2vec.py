import json, re
from gensim.models import Word2Vec
from tqdm import tqdm

def tokenize(func):
    tokens = re.findall(r'[a-zA-Z_]\w*|[{}();,=+\-*/<>!&|]', func)
    return tokens

with open("data/raw/dataset.json") as f:
    data = json.load(f)

corpus = [tokenize(d['func']) for d in tqdm(data)]
print(corpus[0])

model = Word2Vec(
    sentences=corpus,
    vector_size=100,   
    window=5,
    min_count=1,
    workers=4,
    epochs=10
)

model.save("data/embeddings/word2vec.bin")
print("Vocab size:", len(model.wv))