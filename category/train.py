import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from lightgbm import LGBMClassifier
import joblib
import os

# ------------------------------
# 1) VERİYİ YÜKLE
# ------------------------------
CSV_PATH = "../products.csv"
df = pd.read_csv(CSV_PATH)

print(f"Toplam örnek sayısı (ham veri): {len(df)}")

# ------------------------------
# 2) EKSİK VERİ TEMİZLİĞİ
# ------------------------------
df["name"] = df["name"].fillna("")
df["description"] = df["description"].fillna("")
df["text"] = df["name"] + ". " + df["description"]

# ------------------------------
# 3) AZ ÖRNEKLİ KATEGORİLERİ ÇIKAR
# ------------------------------
MIN_SAMPLES = 10
class_counts = df["category"].value_counts()
print(f"Kategori sayısı {class_counts}")
valid_categories = class_counts[class_counts >= MIN_SAMPLES].index
df = df[df["category"].isin(valid_categories)]

print(f"Minimum {MIN_SAMPLES} örnekli kategori sayısı: {len(valid_categories)}")
print(f"Filtre sonrası örnek sayısı: {len(df)}")

# ------------------------------
# 4) LABEL ENCODING
# ------------------------------
le = LabelEncoder()
df["label"] = le.fit_transform(df["category"])

# ------------------------------
# 5) TRAIN / TEST SPLIT
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"].values,
    df["label"].values,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"Eğitim örnekleri: {len(X_train)}, Test örnekleri: {len(X_test)}")

# ------------------------------
# 6) EMBEDDING MODELİ
# ------------------------------
print("Embedding modeli yükleniyor...")
emb_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

print("Embedding çıkarılıyor (train)...")
emb_train = emb_model.encode(X_train, batch_size=64, show_progress_bar=True)

print("Embedding çıkarılıyor (test)...")
emb_test = emb_model.encode(X_test, batch_size=64, show_progress_bar=True)

# ------------------------------
# 7) CLASSIFIER EĞİTİM
# ------------------------------
print("Model eğitiliyor...")
clf = LGBMClassifier(
    num_leaves=64,
    learning_rate=0.05,
    n_estimators=400,
    class_weight="balanced"
)

clf.fit(emb_train, y_train)

# ------------------------------
# 8) DEĞERLENDİRME
# ------------------------------
preds = clf.predict(emb_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")

print("\n✅ Accuracy:", round(acc, 4))
print("✅ Macro F1:", round(f1, 4))

# ------------------------------
# 9) MODELLERİ KAYDET
# ------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(clf, "models/classifier.pkl")
joblib.dump(le, "models/label_encoder.pkl")
emb_model.save("models/embedding_model")

print("\n🎉 Eğitim tamamlandı!")
print("📁 Kaydedilen dosyalar:")
print(" - models/classifier.pkl")
print(" - models/label_encoder.pkl")
print(" - models/embedding_model/")
print("\nBu modeller Flask API ile doğrudan uyumludur ✅")
