# ============================================================
# STEP 1 — INSTALL DEPENDENCIES
# ============================================================
# import subprocess
# import sys

# subprocess.run([sys.executable, "-m", "pip", "install", "-q",
#                 "chromadb", "sentence-transformers",
#                 "rank_bm25", "numpy"], check=True)


# ============================================================
# STEP 2 — CONFIGURATION
# Must match what you used in Layer 1
# ============================================================
CHROMA_PATH   = "./chroma_db"    # same path as Layer 1
REPO_NAME     = "momulaharinathreddy74-cyber/Object_detection-" # same repo as Layer 1
TOP_K_DENSE   = 10    # how many chunks to pull from dense search
TOP_K_BM25    = 10    # how many chunks to pull from BM25 search
TOP_K_FINAL   = 5     # final chunks after re-ranking

print(f"Config set — ChromaDB path: {CHROMA_PATH}")


# ============================================================
# STEP 3 — LOAD EMBEDDING MODEL & RECONNECT TO CHROMADB
# Reload the same model and collections from Layer 1
# ============================================================
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, CrossEncoder

# Embedding model (same as Layer 1)
print("Loading embedding model...")
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Embedding model loaded")

class MiniLMEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input):
        return embed_model.encode(input).tolist()

embed_fn = MiniLMEmbeddingFunction()

# Reconnect to existing ChromaDB collections from Layer 1
client      = chromadb.PersistentClient(path=CHROMA_PATH)
col_code    = client.get_or_create_collection("source_code",   embedding_function=embed_fn)
col_commits = client.get_or_create_collection("git_commits",   embedding_function=embed_fn)
col_issues  = client.get_or_create_collection("github_issues", embedding_function=embed_fn)
col_docs    = client.get_or_create_collection("readme_docs",   embedding_function=embed_fn)

COLLECTIONS = {
    "source_code"   : col_code,
    "git_commits"   : col_commits,
    "github_issues" : col_issues,
    "readme_docs"   : col_docs,
}

print(f"Reconnected to ChromaDB collections:")
for name, col in COLLECTIONS.items():
    print(f"  {name}: {col.count()} chunks")


# ============================================================
# STEP 4 — LOAD CROSS-ENCODER RE-RANKER
# cross-encoder/ms-marco-MiniLM-L-6-v2 is free on HuggingFace
# It scores (query, chunk) pairs and picks the most relevant ones
# ============================================================
print("\nLoading cross-encoder re-ranker...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Re-ranker loaded")


# ============================================================
# STEP 5 — BUILD BM25 INDEX FOR EACH COLLECTION
# BM25 is a keyword-based search algorithm (like TF-IDF)
# Good for exact lookups: function names, error codes, etc.
# ============================================================
from rank_bm25 import BM25Okapi
import re

def tokenize(text):
    """Lowercase and split text into tokens for BM25."""
    return re.findall(r'\w+', text.lower())

def build_bm25_index(collection):
    """
    Fetch all documents from a ChromaDB collection
    and build a BM25 index from them.
    Returns: (bm25_index, list_of_documents, list_of_metadatas)
    """
    all_data  = collection.get(include=["documents", "metadatas"])
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    if not documents:
        return None, [], []
    tokenized = [tokenize(doc) for doc in documents]
    bm25      = BM25Okapi(tokenized)
    return bm25, documents, metadatas

print("\nBuilding BM25 indexes for all collections...")
bm25_indexes = {}
for name, col in COLLECTIONS.items():
    bm25, docs, metas     = build_bm25_index(col)
    bm25_indexes[name]    = {"bm25": bm25, "docs": docs, "metas": metas}
    print(f"  BM25 index built for '{name}' — {len(docs)} documents")
print("BM25 indexes ready")


# ============================================================
# STEP 6 — DENSE RETRIEVAL (Semantic Search via ChromaDB)
# Uses the embedding model to find semantically similar chunks
# e.g. "where is auth handled" -> finds auth-related code
# ============================================================
def dense_search(query, collection, top_k=10):
    """
    Semantic search using ChromaDB embeddings.
    Returns list of dicts with text, metadata, and rank.
    """
    collection_size = collection.count()
    if collection_size == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection_size),
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return [
        {"text": doc, "metadata": meta, "dense_rank": i + 1}
        for i, (doc, meta) in enumerate(zip(documents, metadatas))
    ]


# ============================================================
# STEP 7 — BM25 RETRIEVAL (Keyword Search)
# Uses exact token matching to find precise results
# e.g. "find usages of verify_token" -> finds exact function name
# ============================================================
def bm25_search(query, index_data, top_k=10):
    """
    Keyword search using BM25.
    Returns list of dicts with text, metadata, and rank.
    """
    bm25      = index_data["bm25"]
    docs      = index_data["docs"]
    metas     = index_data["metas"]
    tokens    = tokenize(query)
    if bm25 is None or not docs:
        return []
    scores    = bm25.get_scores(tokens)
    top_idxs  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"text": docs[i], "metadata": metas[i], "bm25_rank": rank + 1, "bm25_score": scores[i]}
        for rank, i in enumerate(top_idxs)
        if scores[i] > 0  # only include results with actual keyword matches
        
    ]


# ============================================================
# STEP 8 — RECIPROCAL RANK FUSION (RRF)
# Merges dense + BM25 results into a single ranked list
# RRF formula: score = sum(1 / (rank + 60)) for each result list
# Higher score = appeared highly ranked in more result lists
# ============================================================
def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    """Fuse dense and BM25 rankings."""
    scores = {}
    chunks = {}
    for result in dense_results:
        key = result["text"]
        scores[key] = scores.get(key, 0) + 1 / (result["dense_rank"] + k)
        chunks[key] = result
    for result in bm25_results:
        key = result["text"]
        scores[key] = scores.get(key, 0) + 1 / (result["bm25_rank"] + k)
        chunks[key] = result

    fused = []
    for key in sorted(scores, key=scores.get, reverse=True):
        result = chunks[key].copy()
        result["rrf_score"] = round(scores[key], 6)
        fused.append(result)
    return fused


# ============================================================
# STEP 9 — CROSS-ENCODER RE-RANKING
# Takes the fused list and re-scores every (query, chunk) pair
# More accurate than embedding similarity — reads both together
# ============================================================
def rerank(query, fused_results, top_k=5):
    """Re-rank fused results with the cross-encoder."""
    if not fused_results:
        return []
    pairs = [(query, result["text"]) for result in fused_results]
    predictions = reranker.predict(pairs)
    ranked = sorted(zip(predictions, fused_results), key=lambda item: item[0], reverse=True)
    final = []
    for score, result in ranked[:top_k]:
        result = result.copy()
        result["rerank_score"] = round(float(score), 4)
        final.append(result)
    return final

# ============================================================
# STEP 10 — FULL HYBRID SEARCH PIPELINE
# Ties everything together: Dense + BM25 -> RRF -> Re-rank
# Can search one collection or all 4 simultaneously
# ============================================================
def hybrid_search(query, collection_names=None, top_k_final=TOP_K_FINAL):
    """Search collections with dense retrieval and BM25, then fuse and re-rank."""
    collection_names = collection_names or list(COLLECTIONS)
    unknown = set(collection_names) - set(COLLECTIONS)
    if unknown:
        raise ValueError(f"Unknown collections: {', '.join(sorted(unknown))}")

    all_dense = []
    all_bm25 = []
    for name in collection_names:
        dense_results = dense_search(query, COLLECTIONS[name], TOP_K_DENSE)
        bm25_results = bm25_search(query, bm25_indexes[name], TOP_K_BM25)
        for result in dense_results:
            result["collection"] = name
        for result in bm25_results:
            result["collection"] = name
        all_dense.extend(dense_results)
        all_bm25.extend(bm25_results)

    for rank, result in enumerate(all_dense, start=1):
        result["dense_rank"] = rank
    for rank, result in enumerate(all_bm25, start=1):
        result["bm25_rank"] = rank

    fused = reciprocal_rank_fusion(all_dense, all_bm25)
    return rerank(query, fused, top_k_final)
# ============================================================
# STEP 11 — PRETTY PRINT RESULTS
# ============================================================
def print_results(query, results):
    print(f"\n{'='*60}")
    print(f" Query : {query}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        meta = r.get("metadata", {})
        print(f"\n  Result {i+1} | Collection: {r.get('collection','?')} | Rerank Score: {r.get('rerank_score', '?')}")
        # Print relevant metadata per source type
        source = meta.get("source", "")
        if source == "source_code":
            print(f"  File : {meta.get('file_path')} | {meta.get('type')}: {meta.get('name')} | Lines {meta.get('start_line')}-{meta.get('end_line')}")
        elif source == "git_commits":
            print(f"  Commit : {meta.get('commit_sha')} | Author: {meta.get('author')} | Date: {meta.get('date')}")
        elif source == "github_issues":
            print(f"  Issue #{meta.get('issue_number')} | {meta.get('title')} | State: {meta.get('state')}")
        elif source == "readme_docs":
            print(f"  Doc File : {meta.get('file_path')} | Chunk {meta.get('chunk_index')}/{meta.get('total_chunks')}")
        print(f"  Preview  : {r['text'][:250]}...")
        print(f"  RRF Score: {r.get('rrf_score','?')}")
    print(f"\n{'='*60}")


# ============================================================
# STEP 12 — RUN A TEST QUERY
# ============================================================
if __name__ == "__main__":
    query = "how does object detection work in this repository"
    results = hybrid_search(
        query,
        collection_names=["source_code", "git_commits"],
        top_k_final=TOP_K_FINAL,
    )
    print_results(query, results)


# ============================================================
# STEP 12 — RUN TEST QUERIES
# Test different query types to show Dense vs BM25 strengths
# ============================================================

# Query 1: Semantic query — Dense search shines here
q1 = "where is authentication and security handled"
r1 = hybrid_search(q1, collection_names=["source_code"], top_k_final=TOP_K_FINAL)
print_results(q1, r1)

# Query 2: Keyword/exact query — BM25 shines here
q2 = "HTTPException status_code 404"
r2 = hybrid_search(q2, collection_names=["source_code", "github_issues"], top_k_final=TOP_K_FINAL)
print_results(q2, r2)

# Query 3: Cross-source query — searches all 4 collections
q3 = "what changed recently and why"
r3 = hybrid_search(q3, collection_names=None, top_k_final=TOP_K_FINAL)
print_results(q3, r3)

# Query 4: Doc query — installation and getting started
q4 = "how to install and get started"
r4 = hybrid_search(q4, collection_names=["readme_docs"], top_k_final=TOP_K_FINAL)
print_results(q4, r4)


# ============================================================
# STEP 13 — COMPARE: DENSE ONLY vs HYBRID
# Shows why hybrid retrieval is better than dense alone
# ============================================================
print("\n" + "="*60)
print("  COMPARISON: Dense Only vs Hybrid Retrieval")
print("="*60)

test_query = "dependency injection"

# Dense only
dense_only = dense_search(test_query, col_code, top_k=3)
print(f"\nDENSE ONLY — Top 3 for '{test_query}':")
for i, r in enumerate(dense_only):
    print(f"  {i+1}. [{r['metadata'].get('name')}] {r['text'][:120]}...")

# Hybrid
hybrid_only = hybrid_search(test_query, collection_names=["source_code"], top_k_final=3)
print(f"\nHYBRID (Dense + BM25 + Rerank) — Top 3 for '{test_query}':")
for i, r in enumerate(hybrid_only):
    print(f"  {i+1}. [{r['metadata'].get('name')}] Score:{r.get('rerank_score')} — {r['text'][:120]}...")

print("\nLayer 2 complete! Hybrid retrieval pipeline is ready.")
print("Next: Layer 3 will add the Agentic Loop with multi-hop reasoning.")
