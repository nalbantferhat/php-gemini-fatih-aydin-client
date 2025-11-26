new_emb = emb_model.encode([new_name]).astype("float32")
index.add(new_emb)
products.append({"code": new_code, "name": new_name})

faiss.write_index(index, "data/products.index")

with open("data/products.json", "w", encoding="utf-8") as f:
    json.dump(products, f, ensure_ascii=False)
