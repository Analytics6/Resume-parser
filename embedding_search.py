import re
import time
from typing import Any

import chromadb
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer

from resume_data import CATEGORIES, example_queries


MODEL_DEFINITIONS = [
    {"name": "TF-IDF", "type": "tfidf"},
    {"name": "CountVectorizer", "type": "count"},
    {"name": "HashingVectorizer", "type": "hashing"},
    {"name": "Word2Vec", "type": "word2vec"},
    {"name": "SentenceTransformer: all-MiniLM-L6-v2", "type": "sentence_transformer", "model": "all-MiniLM-L6-v2"},
    {"name": "SentenceTransformer: all-MiniLM-L12-v2", "type": "sentence_transformer", "model": "all-MiniLM-L12-v2"},
    {"name": "SentenceTransformer: paraphrase-MiniLM-L6-v2", "type": "sentence_transformer", "model": "paraphrase-MiniLM-L6-v2"},
    {"name": "SentenceTransformer: paraphrase-MiniLM-L12-v2", "type": "sentence_transformer", "model": "paraphrase-MiniLM-L12-v2"},
    {"name": "SentenceTransformer: all-mpnet-base-v2", "type": "sentence_transformer", "model": "all-mpnet-base-v2"},
    {"name": "SentenceTransformer: BAAI/bge-small-en-v1.5", "type": "sentence_transformer", "model": "BAAI/bge-small-en-v1.5"},
]

MODEL_CACHE: dict[str, Any] = {}


def _collection_name_for(model: dict) -> str:
    return "resume_" + re.sub(r"[^a-z0-9]+", "_", model["name"].lower()).strip("_")


def _tokenize(text: str):
    return re.findall(r"\b[a-zA-Z0-9-]+\b", text.lower())


def _fit_or_get_model(model: dict, texts):
    key = model["name"]
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    if model["type"] == "tfidf":
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        matrix = vectorizer.fit_transform(texts)
        trained = {"vectorizer": vectorizer, "matrix": matrix}
        MODEL_CACHE[key] = trained
        return trained

    if model["type"] == "count":
        vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        matrix = vectorizer.fit_transform(texts)
        trained = {"vectorizer": vectorizer, "matrix": matrix}
        MODEL_CACHE[key] = trained
        return trained

    if model["type"] == "hashing":
        vectorizer = HashingVectorizer(n_features=2**14, alternate_sign=False, stop_words="english")
        matrix = vectorizer.transform(texts)
        trained = {"vectorizer": vectorizer, "matrix": matrix}
        MODEL_CACHE[key] = trained
        return trained

    if model["type"] == "word2vec":
        sentences = [_tokenize(text) for text in texts]
        model_w2v = Word2Vec(sentences=sentences, vector_size=100, min_count=1, workers=1, epochs=25, seed=42)
        MODEL_CACHE[key] = model_w2v
        return model_w2v

    if model["type"] == "sentence_transformer":
        st_model = SentenceTransformer(model["model"])
        MODEL_CACHE[key] = st_model
        return st_model

    raise ValueError(f"Unsupported model type: {model['type']}")


def _collect_embeddings(model, texts):
    trained = _fit_or_get_model(model, texts)

    if model["type"] in {"tfidf", "count", "hashing"}:
        vectorizer = trained["vectorizer"]
        matrix = vectorizer.transform(texts)
        return np.asarray(matrix.toarray(), dtype=np.float32), vectorizer

    if model["type"] == "word2vec":
        sentences = [_tokenize(text) for text in texts]
        vectors = []
        for tokens in sentences:
            token_vectors = [trained.wv[token] for token in tokens if token in trained.wv]
            if len(token_vectors) == 0:
                vectors.append(np.zeros(100, dtype=np.float32))
            else:
                vectors.append(np.mean(token_vectors, axis=0).astype(np.float32))
        return np.asarray(vectors, dtype=np.float32), trained

    if model["type"] == "sentence_transformer":
        embeddings = trained.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(embeddings, dtype=np.float32), trained

    raise ValueError(f"Unsupported model type: {model['type']}")


def _query_embedding_for_model(model: dict, query: str):
    trained = _fit_or_get_model(model, [query])

    if model["type"] in {"tfidf", "count", "hashing"}:
        vectorizer = trained["vectorizer"]
        return np.asarray(vectorizer.transform([query]).toarray()[0], dtype=np.float32)

    if model["type"] == "word2vec":
        tokens = _tokenize(query)
        if not tokens:
            return np.zeros(100, dtype=np.float32)
        vecs = [trained.wv[token] for token in tokens if token in trained.wv]
        if not vecs:
            return np.zeros(100, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    if model["type"] == "sentence_transformer":
        return np.asarray(trained.encode([query], show_progress_bar=False), dtype=np.float32)[0]

    raise ValueError(f"Unsupported model type: {model['type']}")


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_chroma_collections(df: pd.DataFrame):
    client = chromadb.PersistentClient(path="./chroma_store")
    results = []
    for model in MODEL_DEFINITIONS:
        texts = df["resume_text"].tolist()
        embeddings, _ = _collect_embeddings(model, texts)
        collection_name = _collection_name_for(model)
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        ids = [f"resume_{idx}" for idx in df.index]
        metadatas = [{"category": row["category"], "title": row["title"], "resume_id": row["resume_id"]} for _, row in df.iterrows()]
        collection.add(documents=texts, embeddings=embeddings.tolist(), ids=ids, metadatas=metadatas)
        results.append({"model": model["name"], "collection": collection, "model_def": model})
    return results


def search_collection(collection, query: str, n_results: int = 5):
    model_key = collection.name
    matching_model = next((m for m in MODEL_DEFINITIONS if _collection_name_for(m) == model_key), None)
    if matching_model is None:
        raise ValueError(f"No matching embedding model found for collection {model_key}")

    if matching_model["name"] not in MODEL_CACHE:
        raise ValueError(f"Model {matching_model['name']} has not been trained yet. Run build_chroma_collections() first.")

    query_vector = _query_embedding_for_model(matching_model, query).tolist()
    result = collection.query(query_embeddings=[query_vector], n_results=n_results, include=["documents", "metadatas", "distances"])
    return result


def compare_models(df: pd.DataFrame, top_n: int = 5):
    client = chromadb.PersistentClient(path="./chroma_store")
    query_map = example_queries()
    report_rows = []
    for model in MODEL_DEFINITIONS:
        name = model["name"]
        collection_name = _collection_name_for(model)
        collection = client.get_collection(name=collection_name)
        hits = 0
        total = 0
        avg_latency = 0.0
        for expected_category, query in query_map.items():
            start = time.perf_counter()
            result = search_collection(collection, query, n_results=top_n)
            latency_ms = (time.perf_counter() - start) * 1000
            avg_latency += latency_ms
            metadata = result.get("metadatas", [[]])[0]
            categories = [item.get("category", "Unknown") for item in metadata]
            matches = 1 if expected_category in categories else 0
            hits += matches
            total += 1
        score = hits / total if total else 0
        report_rows.append({
            "model": name,
            "category_match_rate": round(score * 100, 2),
            "average_latency_ms": round(avg_latency / total, 2) if total else 0,
            "n_queries": total,
        })
    return pd.DataFrame(report_rows).sort_values("category_match_rate", ascending=False)


def get_query_results(df: pd.DataFrame, query: str, top_n: int = 5):
    results = []
    client = chromadb.PersistentClient(path="./chroma_store")
    for model in MODEL_DEFINITIONS:
        name = model["name"]
        collection_name = _collection_name_for(model)
        collection = client.get_collection(name=collection_name)
        query_response = search_collection(collection, query, n_results=top_n)
        documents = query_response.get("documents", [[]])[0]
        metadatas = query_response.get("metadatas", [[]])[0]
        distances = query_response.get("distances", [[]])[0]
        results.append({
            "model": name,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        })
    return results
