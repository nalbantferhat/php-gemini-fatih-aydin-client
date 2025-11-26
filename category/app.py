from flask import Flask, request, jsonify
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

# ------------------------------
# MODELLERİ YÜKLE
# ------------------------------
CLASSIFIER_PATH = "models/classifier.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
EMBEDDING_MODEL_PATH = "models/embedding_model"

print("Modeller yükleniyor...")
clf = joblib.load(CLASSIFIER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
emb_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
print("Modeller yüklendi.")

# ------------------------------
# FLASK APP
# ------------------------------
app = Flask(__name__)


def predict_top1(name: str, description: str):
    """
    Tek bir ürün için EN İYİ kategoriyi ve confidence skorunu döndürür.
    """
    text = f"{name}. {description}"
    emb = emb_model.encode([text])  # (1, embedding_dim)

    # Probabilistic classifier ise:
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(emb)[0]  # (num_classes,)
        best_idx = np.argmax(probs)
        best_category = le.inverse_transform([best_idx])[0]
        best_confidence = float(probs[best_idx])
        return best_category, best_confidence
    else:
        # predict_proba yoksa fallback
        pred_label = clf.predict(emb)[0]
        best_category = le.inverse_transform([pred_label])[0]
        return best_category, 1.0


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON INPUT:
    {
      "name": "Kot etek",
      "description": "Yüksek bel dar kesim kadın eteği"
    }

    JSON OUTPUT:
    {
      "category": "121",
      "confidence": 0.82
    }
    """
    try:
        data = request.get_json(force=True)

        name = data.get("name", "")
        description = data.get("description", "")

        if not isinstance(name, str) or not isinstance(description, str):
            return jsonify({"error": "name ve description string olmalı"}), 400

        category, confidence = predict_top1(name, description)

        return jsonify({
            "category": str(category),
            "confidence": confidence
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
