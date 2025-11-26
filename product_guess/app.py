from flask import Flask, request, jsonify
import ujson as json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------
# 1) MODELLERİ YÜKLE
# ----------------------------
print("Modeller yükleniyor...")
emb_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

index = faiss.read_index("data/products.index")

with open("data/products.json", "r", encoding="utf-8") as f:
    products = json.load(f)

print("✅ API hazır.")

# ----------------------------
# FLASK UYGULAMASI
# ----------------------------
app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    """
    INPUT:
    {
      "query": "manyetik röle"
    }

    OUTPUT:
    {
      "results": [
        {"code": "...", "name": "...", "score": 0.91},
        ...
      ]
    }
    """
    data = request.get_json(force=True)
    query = data.get("query", "")

    # Embed query
    q_emb = emb_model.encode([query]).astype("float32")

    # Top-3 arama
    distances, ids = index.search(q_emb, 3)

    results = []
    for rank, idx in enumerate(ids[0]):
        product = products[idx]
        score = float(distances[0][rank])
        results.append({
            "code": product["code"],
            "name": product["name"],
            "score": score
        })

    return jsonify({"results": results}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=True)
