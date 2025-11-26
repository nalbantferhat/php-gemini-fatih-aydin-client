import ujson as json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------
# 1) ÜRÜN LİSTESİNİ YÜKLE
# ----------------------------
with open("data/products.json", "r", encoding="utf-8") as f:
    products = json.load(f)

names = [p["name"] for p in products]

# ----------------------------
# 2) EMBEDDING MODELİ
# ----------------------------
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

print("Embedding çıkarılıyor...")
emb = model.encode(names, batch_size=64, show_progress_bar=True).astype("float32")

dim = emb.shape[1]  # 768

# ----------------------------
# 3) HNSW INDEX
# ----------------------------
index = faiss.IndexHNSWFlat(dim, 32)
index.hnsw.efConstruction = 200

print("Index oluşturuluyor...")
index.add(emb)

# ----------------------------
# 4) KAYDET
# ----------------------------
faiss.write_index(index, "data/products.index")

print("✅ Index hazır!")
print(" - data/products.index")
print(" - data/products.json")
