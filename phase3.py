# ============================================================
# STEP 1 — INSTALL DEPENDENCIES
# ============================================================
# import importlib.util
# import subprocess
# import sys
import os
from dotenv import load_dotenv

load_dotenv()
# REQUIRED_PACKAGES = {
#     "chromadb": "chromadb",
#     "sentence_transformers": "sentence-transformers",
#     "rank_bm25": "rank-bm25",
#     "requests": "requests",
#     "numpy": "numpy",
# }

# missing_packages = [
#     package
#     for module, package in REQUIRED_PACKAGES.items()
#     if importlib.util.find_spec(module) is None
# ]

# if missing_packages:
#     print(f"Installing missing packages: {', '.join(missing_packages)}")
#     subprocess.run(
#         [sys.executable, "-m", "pip", "install", *missing_packages],
#         check=True,
#     )
# else:
#     print("All required packages are already installed.")


# ============================================================
# STEP 2 — CONFIGURATION
# HF Token: huggingface.co -> Settings -> Access Tokens -> New Token (Read)
# ============================================================
CHROMA_PATH   = "./chroma_db"      # same path as Layer 1 & 2
REPO_NAME     = "momulaharinathreddy74-cyber/Object_detection-"            # same repo as Layer 1 & 2
HF_TOKEN = os.getenv("HF_TOKEN")       # <-- paste your HF token here
TOP_K_DENSE   = 10
TOP_K_BM25    = 10
TOP_K_FINAL   = 5
MAX_HOPS      = 3    # maximum retrieval hops the agent can take
MAX_SUBQ      = 3    # maximum sub-questions to decompose into

# Hugging Face Inference Providers — OpenAI-compatible chat endpoint
HF_API_URL    = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL      = "Qwen/Qwen2.5-7B-Instruct"
HF_HEADERS    = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

print(f"Config set | Repo: {REPO_NAME} | Max hops: {MAX_HOPS}")


# ============================================================
# STEP 3 — RELOAD EMBEDDING MODEL & CHROMADB (from Layer 1)
# ============================================================
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, CrossEncoder

print("Loading models...")
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
reranker    = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Models loaded")

class MiniLMEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        return embed_model.encode(input).tolist()

embed_fn    = MiniLMEmbeddingFunction()
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
print("ChromaDB reconnected")


# ============================================================
# STEP 4 — REBUILD BM25 INDEXES (from Layer 2)
# ============================================================
from rank_bm25 import BM25Okapi
import re

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def build_bm25_index(collection):
    all_data  = collection.get(include=["documents", "metadatas"])
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    if not documents:
        return None, [], []
    bm25      = BM25Okapi([tokenize(doc) for doc in documents])
    return bm25, documents, metadatas

print("Building BM25 indexes...")
bm25_indexes = {}
for name, col in COLLECTIONS.items():
    bm25, docs, metas  = build_bm25_index(col)
    bm25_indexes[name] = {"bm25": bm25, "docs": docs, "metas": metas}
print("BM25 indexes ready")


# ============================================================
# STEP 5 — HYBRID SEARCH PIPELINE (from Layer 2)
# Dense + BM25 + RRF Fusion + Cross-Encoder Rerank
# ============================================================
def dense_search(query, collection, top_k=10):
    n       = min(top_k, collection.count())
    if n == 0:
        return []
    results = collection.query(query_texts=[query], n_results=n)
    return [
        {"text": doc, "metadata": meta, "dense_rank": i + 1}
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]))
    ]

def bm25_search(query, index_data, top_k=10):
    if index_data["bm25"] is None:
        return []
    scores   = index_data["bm25"].get_scores(tokenize(query))
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"text": index_data["docs"][i], "metadata": index_data["metas"][i],
         "bm25_rank": rank + 1, "bm25_score": scores[i]}
        for rank, i in enumerate(top_idxs) if scores[i] > 0
    ]

def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    scores, all_chunks = {}, {}
    for r in dense_results:
        key = r["text"][:100]
        scores[key]     = scores.get(key, 0) + 1 / (r["dense_rank"] + k)
        all_chunks[key] = r
    for r in bm25_results:
        key = r["text"][:100]
        scores[key]     = scores.get(key, 0) + 1 / (r["bm25_rank"] + k)
        all_chunks[key] = r
    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [{**all_chunks[k], "rrf_score": round(scores[k], 6)} for k in sorted_keys]

def rerank(query, fused_results, top_k=5):
    if not fused_results:
        return []
    scores = reranker.predict([(query, r["text"]) for r in fused_results])
    ranked = sorted(zip(scores, fused_results), key=lambda x: x[0], reverse=True)
    return [{**r, "rerank_score": round(float(s), 4)} for s, r in ranked[:top_k]]

def hybrid_search(query, collection_names=None, top_k_final=5):
    if collection_names is None:
        collection_names = list(COLLECTIONS.keys())
    all_dense, all_bm25 = [], []
    for name in collection_names:
        d = dense_search(query, COLLECTIONS[name], top_k=TOP_K_DENSE)
        b = bm25_search(query, bm25_indexes[name], top_k=TOP_K_BM25)
        for r in d: r["collection"] = name
        for r in b: r["collection"] = name
        all_dense.extend(d)
        all_bm25.extend(b)
    for i, r in enumerate(all_dense): r["dense_rank"] = i + 1
    for i, r in enumerate(all_bm25):  r["bm25_rank"]  = i + 1
    fused = reciprocal_rank_fusion(all_dense, all_bm25)
    return rerank(query, fused, top_k=top_k_final)

print("Hybrid search pipeline ready")


# ============================================================
# STEP 6 — LLM CALL via Hugging Face Inference Providers
# Uses a currently supported instruction model through the chat API
# ============================================================
import requests
import json

def call_llm(prompt, max_new_tokens=512):
    """
    Call the Hugging Face chat-completions API.
    Returns the generated text string.
    """
    if not HF_TOKEN:
        return "[LLM Error]: HF_TOKEN is missing from the .env file"

    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
    }
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        result   = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        error_detail = response.text[:500] if "response" in locals() else str(e)
        return f"[LLM call failed]: {error_detail}"
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return f"[LLM call failed]: {e}"


# ============================================================
# STEP 7 — QUESTION DECOMPOSITION
# The agent breaks a complex question into simpler sub-questions
# e.g. "Why was verify_token changed?" ->
#      ["What does verify_token do?", "Which commits touched verify_token?",
#       "What issues triggered those commits?"]
# ============================================================
def decompose_question(question):
    """
    Use LLM to break the user question into sub-questions.
    Each sub-question will be retrieved separately (multi-hop).
    """
    prompt = f"""<s>[INST]
You are a code intelligence assistant analyzing a GitHub repository.
Break the following question into at most {MAX_SUBQ} simpler, specific sub-questions
that can each be answered by searching code, commits, issues, or documentation.

Question: {question}

Return ONLY a numbered list of sub-questions, nothing else. Example:
1. What does X function do?
2. Which commits modified X?
3. What issues mentioned X?
[/INST]"""

    response = call_llm(prompt, max_new_tokens=200)

    # Parse numbered list from LLM response
    sub_questions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            q = line.split(".", 1)[1].strip()
            if q:
                sub_questions.append(q)

    # Fallback: if LLM didn't return a proper list, use original question
    if not sub_questions:
        sub_questions = [question]

    return sub_questions[:MAX_SUBQ]


# ============================================================
# STEP 8 — REFERENCE FOLLOWER
# When a chunk mentions a function name, file, or issue number,
# the agent follows that reference to fetch related chunks.
# This is what enables multi-hop reasoning.
# e.g. code chunk mentions "verify_token" -> search commits for "verify_token"
# ============================================================
def extract_references(chunks):
    """
    Extract function names, file paths, and issue numbers
    from retrieved chunks to follow as next-hop queries.
    """
    references = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        text = chunk.get("text", "")

        # Extract function/class name from code chunks
        if meta.get("source") == "source_code" and meta.get("name"):
            references.append({"type": "function", "value": meta["name"]})

        # Extract file paths from code chunks
        if meta.get("file_path"):
            references.append({"type": "file", "value": meta["file_path"]})

        # Extract issue numbers from text (e.g. #123)
        issue_refs = re.findall(r'#(\d+)', text)
        for num in issue_refs[:2]:  # limit to 2 issue refs per chunk
            references.append({"type": "issue", "value": num})

    # Deduplicate
    seen = set()
    unique = []
    for ref in references:
        key = f"{ref['type']}:{ref['value']}"
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    return unique[:5]  # cap at 5 references per hop


def follow_references(refs, already_seen_texts):
    """
    For each extracted reference, run a targeted search
    in the most relevant collection.
    Returns new chunks not already seen.
    """
    new_chunks = []
    for ref in refs:
        if ref["type"] == "function":
            # Search commits and issues that mention this function
            results = hybrid_search(
                ref["value"],
                collection_names=["git_commits", "github_issues"],
                top_k_final=2
            )
        elif ref["type"] == "file":
            # Search commits that touched this file
            results = hybrid_search(
                ref["value"],
                collection_names=["git_commits"],
                top_k_final=2
            )
        elif ref["type"] == "issue":
            # Search for this specific issue
            results = hybrid_search(
                f"issue {ref['value']}",
                collection_names=["github_issues"],
                top_k_final=2
            )
        else:
            continue

        # Only add chunks we haven't seen before
        for r in results:
            if r["text"][:100] not in already_seen_texts:
                new_chunks.append(r)
                already_seen_texts.add(r["text"][:100])

    return new_chunks


# ============================================================
# STEP 9 — AGENTIC MULTI-HOP RETRIEVAL LOOP
# This is the core of Layer 3.
# The agent:
#   1. Decomposes question into sub-questions
#   2. Retrieves for each sub-question (Hop 1)
#   3. Extracts references from results
#   4. Follows references to fetch more context (Hop 2, 3...)
#   5. Stops when enough context is gathered or MAX_HOPS reached
# ============================================================
def agentic_retrieval(question):
    """
    Multi-hop retrieval agent.
    Returns: (all_chunks, reasoning_trace)
      - all_chunks     : every chunk gathered across all hops
      - reasoning_trace: step-by-step log of what the agent did
    """
    print(f"\nAgent starting on: '{question}'")
    all_chunks     = []
    seen_texts     = set()
    reasoning_trace = []

    # --- Decompose question into sub-questions ---
    print("  Decomposing question...")
    sub_questions = decompose_question(question)
    reasoning_trace.append({
        "step"    : "decomposition",
        "output"  : sub_questions
    })
    print(f"  Sub-questions: {sub_questions}")

    # --- HOP 1: Retrieve for each sub-question ---
    print("  Hop 1: Retrieving for each sub-question...")
    hop1_chunks = []
    for subq in sub_questions:
        results = hybrid_search(subq, collection_names=None, top_k_final=TOP_K_FINAL)
        for r in results:
            if r["text"][:100] not in seen_texts:
                hop1_chunks.append(r)
                seen_texts.add(r["text"][:100])

    all_chunks.extend(hop1_chunks)
    reasoning_trace.append({
        "step"    : "hop_1_retrieval",
        "queries" : sub_questions,
        "chunks_found": len(hop1_chunks)
    })
    print(f"  Hop 1 done — {len(hop1_chunks)} chunks found")

    # --- HOP 2+: Follow references from hop 1 results ---
    current_chunks = hop1_chunks
    for hop_num in range(2, MAX_HOPS + 1):
        refs = extract_references(current_chunks)
        if not refs:
            print(f"  Hop {hop_num}: No references found — stopping early")
            reasoning_trace.append({"step": f"hop_{hop_num}", "output": "no references found"})
            break

        print(f"  Hop {hop_num}: Following {len(refs)} references...")
        new_chunks = follow_references(refs, seen_texts)

        reasoning_trace.append({
            "step"         : f"hop_{hop_num}_retrieval",
            "references"   : [f"{r['type']}:{r['value']}" for r in refs],
            "chunks_found" : len(new_chunks)
        })

        if not new_chunks:
            print(f"  Hop {hop_num}: No new chunks — stopping")
            break

        all_chunks.extend(new_chunks)
        current_chunks = new_chunks
        print(f"  Hop {hop_num} done — {len(new_chunks)} new chunks")

    print(f"  Retrieval complete — {len(all_chunks)} total chunks across {hop_num} hops")
    return all_chunks, reasoning_trace


# ============================================================
# STEP 10 — CITATION BUILDER
# Every chunk gets a citation tag so the LLM can reference it.
# This prevents hallucination — LLM can only cite what we retrieved.
# ============================================================
def build_context_with_citations(chunks):
    """
    Format all chunks into a numbered context block.
    Each chunk gets a [SOURCE-N] tag the LLM uses to cite.
    Returns: (context_string, citations_map)
    """
    context_parts = []
    citations_map = {}

    for i, chunk in enumerate(chunks):
        meta   = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        tag    = f"SOURCE-{i+1}"

        # Build citation label based on source type
        if source == "source_code":
            label = f"{meta.get('type','')}: {meta.get('name','')} in {meta.get('file_path','')}"
        elif source == "git_commits":
            label = f"Commit {meta.get('commit_sha','')} by {meta.get('author','')} on {meta.get('date','')}"
        elif source == "github_issues":
            label = f"Issue #{meta.get('issue_number','')} — {meta.get('title','')}"
        elif source == "readme_docs":
            label = f"Docs: {meta.get('file_path','')}"
        else:
            label = source

        citations_map[tag] = label
        context_parts.append(
            f"[{tag}] ({label})\n{chunk['text'][:600]}\n"
        )

    return "\n".join(context_parts), citations_map


# ============================================================
# STEP 11 — LLM SYNTHESIS WITH CITATIONS
# The LLM reads all retrieved chunks and produces a cited answer.
# It is instructed to ONLY use information from the context.
# ============================================================
def synthesize_answer(question, chunks, reasoning_trace):
    """
    Feed all retrieved chunks to the LLM and get a cited answer.
    """
    context, citations_map = build_context_with_citations(chunks)

    prompt = f"""<s>[INST]
You are a GitHub codebase intelligence agent. Answer the question using ONLY
the context provided below. For every claim you make, cite the source using
its [SOURCE-N] tag. If the context does not contain enough information,
say so clearly — do NOT make up information.

QUESTION: {question}

CONTEXT:
{context}

Provide a clear, structured answer with citations like [SOURCE-1], [SOURCE-2], etc.
[/INST]"""

    print("  Synthesizing answer with LLM...")
    answer = call_llm(prompt, max_new_tokens=600)
    return answer, citations_map


# ============================================================
# STEP 12 — MAIN AGENT FUNCTION
# Single entry point that runs the full pipeline:
# Question -> Decompose -> Multi-hop Retrieve -> Cite -> Synthesize
# ============================================================
def ask_agent(question):
    """
    Ask the GitHub Intelligence Agent any question about the repo.
    Returns a cited, grounded answer.
    """
    print("\n" + "="*60)
    print(f" QUESTION: {question}")
    print("="*60)

    # Step A: Multi-hop retrieval
    all_chunks, reasoning_trace = agentic_retrieval(question)

    if not all_chunks:
        print("  No relevant chunks found.")
        return

    # Step B: Synthesize cited answer
    answer, citations_map = synthesize_answer(question, all_chunks, reasoning_trace)

    # Step C: Print final answer
    print("\n" + "-"*60)
    print(" ANSWER:")
    print("-"*60)
    print(answer)

    # Step D: Print citation legend
    print("\n" + "-"*60)
    print(" CITATIONS:")
    print("-"*60)
    for tag, label in citations_map.items():
        print(f"  [{tag}] -> {label}")

    # Step E: Print reasoning trace
    print("\n" + "-"*60)
    print(" REASONING TRACE (how the agent retrieved):")
    print("-"*60)
    for step in reasoning_trace:
        print(f"  {step}")

    print("\n" + "="*60)
    return answer


# ============================================================
# STEP 13 — RUN TEST QUESTIONS
# These demonstrate the 3 key agent capabilities:
# 1. Semantic understanding  2. Cross-source linking  3. Multi-hop
# ============================================================

# Question 1: Semantic — finds auth-related code and explains it
ask_agent("Where is authentication handled and how does it work?")

# Question 2: Cross-source — links code to commits to issues
ask_agent("What recently changed in the routing logic and why?")

# Question 3: Multi-hop — 3 hops: code -> commits -> issues
ask_agent("Why was the dependency injection system refactored and what problems did it fix?")

print("\nLayer 3 complete! The agentic loop with multi-hop retrieval is ready.")
print("Next: Layer 4 — MCP Tools (file navigation, live commit fetch, doc creation)")
