from pathlib import Path

import pandas as pd
import streamlit as st

from embedding_search import MODEL_DEFINITIONS, build_chroma_collections, compare_models, get_query_results
from resume_data import example_queries, generate_resume_dataset

st.set_page_config(page_title="Resume Search Benchmark", layout="wide")


@st.cache_data
def load_or_generate_dataset():
    dataset_path = Path("data/resumes.csv")
    if not dataset_path.exists():
        df = generate_resume_dataset(num_per_category=100)
    else:
        df = pd.read_csv(dataset_path)
    return df


@st.cache_resource
def build_collections():
    df = load_or_generate_dataset()
    return build_chroma_collections(df)


st.title("Resume Search and Embedding Comparison")
st.caption("Synthetic benchmark of 100 resumes per category across 10 categories and 10 embedding models.")

with st.sidebar:
    st.header("Dataset")
    st.metric("Categories", 10)
    st.metric("Resumes per category", 100)
    st.metric("Total resumes", 1000)
    st.write("The app uses a synthetic dataset with two-paragraph resumes and indexes them in ChromaDB for semantic search.")

    st.header("Model set")
    st.write("• TF-IDF\n• CountVectorizer\n• HashingVectorizer\n• Word2Vec\n• 6 SentenceTransformers (MiniLM, MPNet, paraphrase, STS-B, e5-base, bge-small)")

    st.header("Example queries")
    query_map = example_queries()
    for category, query in query_map.items():
        if st.button(f"Load: {category}"):
            st.session_state["query"] = query


if "query" not in st.session_state:
    st.session_state["query"] = example_queries()["Software Engineer"]

query = st.text_input("Search query", value=st.session_state["query"])

if st.button("Run search benchmark"):
    st.session_state["query"] = query

try:
    df = load_or_generate_dataset()
    dataset = df.copy()
    collection_status = build_collections()
    comparison = compare_models(dataset)

    st.subheader("Embedding comparison")
    st.dataframe(comparison, use_container_width=True)

    st.subheader("Query: " + query)
    search_results = get_query_results(dataset, query, top_n=5)
    for result in search_results:
        model_name = result["model"]
        docs = result["documents"]
        metas = result["metadatas"]
        distances = result["distances"]
        st.markdown(f"### {model_name}")
        for doc, meta, dist in zip(docs, metas, distances):
            cat = meta.get("category", "Unknown") if isinstance(meta, dict) else "Unknown"
            st.write(f"Category: {cat} | score: {round((1 - dist), 4) if dist is not None else 'n/a'}")
            st.write(doc[:500] + ("..." if len(doc) > 500 else ""))
            st.divider()
except Exception as exc:  # pragma: no cover
    st.error(f"The app hit an error while building the demo: {exc}")
    st.code(str(exc))

st.markdown("---")
st.write("This benchmark is meant for demonstration and comparison, not as a production-grade recruiter recommendation system.")
