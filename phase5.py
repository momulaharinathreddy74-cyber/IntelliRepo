# ============================================================
# STEP 1 — INSTALL DEPENDENCIES
# ============================================================
# import subprocess
# subprocess.run(["pip", "install", "-q",
#                 "chromadb", "sentence-transformers",
#                 "rank_bm25", "PyGithub", "requests",
#                 "numpy", "rich"], check=True)


# ============================================================
# STEP 2 — CONFIGURATION
# ============================================================
import os
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH    = "./chroma_db"    # same as all layers
REPO_NAME      = "momulaharinathreddy74-cyber/Object_detection-"         # same as all layers
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN", "")     # same GitHub token
HF_TOKEN       =os.getenv("HF_TOKEN")          # same HF token
DOC_OUTPUT     = "./generated_docs"  # same as Layer 4
STATE_FILE     = "./index_state.json" # tracks last indexed state
EVAL_REPORT    = "./eval_report.json" # evaluation results saved here
TOP_K_FINAL    = 5
TOP_K_DENSE    = 10
TOP_K_BM25     = 10
MAX_TOOL_CALLS = 5

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

import os, json, re, uuid, datetime
from itertools import islice
os.makedirs(DOC_OUTPUT, exist_ok=True)
print(f"Config set | Repo: {REPO_NAME}")


# ============================================================
# STEP 3 — RELOAD ALL MODELS, CHROMADB & GITHUB (from Layers 1-4)
# ============================================================
import chromadb, requests, numpy as np
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, CrossEncoder
from github import Auth, Github
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

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
    "source_code": col_code, "git_commits": col_commits,
    "github_issues": col_issues, "readme_docs": col_docs
}

g    = Github(auth=Auth.Token(GITHUB_TOKEN))
repo = g.get_repo(REPO_NAME)
print(f"GitHub connected: {repo.full_name}")

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def build_bm25_index(collection):
    all_data = collection.get(include=["documents", "metadatas"])
    docs, metas = all_data["documents"], all_data["metadatas"]
    if not docs:
        return None, [], []
    return BM25Okapi([tokenize(d) for d in docs]), docs, metas

print("Building BM25 indexes...")
bm25_indexes = {}
for name, col in COLLECTIONS.items():
    bm25, docs, metas  = build_bm25_index(col)
    bm25_indexes[name] = {"bm25": bm25, "docs": docs, "metas": metas}
print("All models and indexes ready")


# ============================================================
# STEP 4 — HYBRID SEARCH PIPELINE (carried from Layer 2)
# ============================================================
def dense_search(query, collection, top_k=10):
    n = min(top_k, collection.count())
    if n == 0:
        return []
    r = collection.query(query_texts=[query], n_results=n)
    return [{"text": d, "metadata": m, "dense_rank": i+1}
            for i,(d,m) in enumerate(zip(r["documents"][0], r["metadatas"][0]))]

def bm25_search(query, index_data, top_k=10):
    if index_data["bm25"] is None:
        return []
    scores   = index_data["bm25"].get_scores(tokenize(query))
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"text": index_data["docs"][i], "metadata": index_data["metas"][i],
             "bm25_rank": rank+1, "bm25_score": scores[i]}
            for rank,i in enumerate(top_idxs) if scores[i] > 0]

def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    scores, all_chunks = {}, {}
    for r in dense_results:
        key = r["text"][:100]
        scores[key] = scores.get(key, 0) + 1/(r["dense_rank"]+k)
        all_chunks[key] = r
    for r in bm25_results:
        key = r["text"][:100]
        scores[key] = scores.get(key, 0) + 1/(r["bm25_rank"]+k)
        all_chunks[key] = r
    return [{**all_chunks[k], "rrf_score": round(scores[k],6)}
            for k in sorted(scores, key=lambda k: scores[k], reverse=True)]

def rerank(query, fused, top_k=5):
    if not fused: return []
    scores = reranker.predict([(query, r["text"]) for r in fused])
    ranked = sorted(zip(scores, fused), key=lambda x: x[0], reverse=True)
    return [{**r, "rerank_score": round(float(s),4)} for s,r in ranked[:top_k]]

def hybrid_search(query, collection_names=None, top_k_final=5):
    if collection_names is None:
        collection_names = list(COLLECTIONS.keys())
    all_dense, all_bm25 = [], []
    for name in collection_names:
        d = dense_search(query, COLLECTIONS[name], top_k=TOP_K_DENSE)
        b = bm25_search(query, bm25_indexes[name], top_k=TOP_K_BM25)
        for r in d: r["collection"] = name
        for r in b: r["collection"] = name
        all_dense.extend(d); all_bm25.extend(b)
    for i,r in enumerate(all_dense): r["dense_rank"] = i+1
    for i,r in enumerate(all_bm25):  r["bm25_rank"]  = i+1
    return rerank(query, reciprocal_rank_fusion(all_dense, all_bm25), top_k=top_k_final)


# ============================================================
# STEP 5 — LLM CALL (carried from Layer 3)
# ============================================================
def call_llm(prompt, max_new_tokens=600):
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
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        error_detail = response.text[:500] if "response" in locals() else str(e)
        return f"[LLM call failed]: {error_detail}"
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return f"[LLM call failed]: {e}"


# ============================================================
# STEP 6 — INCREMENTAL INDEXING
# Detects what changed in the repo since last index run
# and only re-indexes new/changed content — not everything.
# This keeps the vector store fresh without full re-ingestion.
# ============================================================
def load_index_state():
    """Load the saved state of what was last indexed."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_commit_sha": None, "last_issue_number": None,
            "indexed_files": [], "last_run": None}

def save_index_state(state):
    """Save current index state to disk."""
    state["last_run"] = str(datetime.datetime.now())
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def check_staleness():
    """
    Compare current repo state to last indexed state.
    Returns a staleness report showing what has changed.
    """
    state = load_index_state()
    report = {"is_stale": False, "new_commits": [], "new_issues": [], "summary": ""}

    # Check new commits
    latest_commits = list(islice(repo.get_commits(), 10))
    if state["last_commit_sha"]:
        new_commits = []
        for c in latest_commits:
            if c.sha[:8] == state["last_commit_sha"]:
                break
            new_commits.append({"sha": c.sha[:8], "message": c.commit.message[:80]})
        report["new_commits"] = new_commits
        if new_commits:
            report["is_stale"] = True
    else:
        report["summary"] = "No previous index state found — full index needed"
        return report

    # Check new issues
    latest_issues = list(islice(
        repo.get_issues(state='all', sort='created', direction='desc'), 5
    ))
    if state["last_issue_number"] and latest_issues:
        new_issues = [i for i in latest_issues if i.number > state["last_issue_number"]]
        report["new_issues"] = [{"number": i.number, "title": i.title} for i in new_issues]
        if new_issues:
            report["is_stale"] = True

    report["summary"] = (
        f"{len(report['new_commits'])} new commits, "
        f"{len(report['new_issues'])} new issues since last index"
    )
    return report

def incremental_reindex():
    """
    Re-index only new commits and issues since last run.
    Does NOT re-ingest all source code (only if files changed).
    """
    print("\nChecking for staleness...")
    report = check_staleness()
    print(f"  Staleness check: {report['summary']}")

    if not report["is_stale"]:
        print("  Index is up to date — no re-indexing needed")
        return report

    state = load_index_state()

    # Re-index new commits only
    if report["new_commits"]:
        print(f"  Re-indexing {len(report['new_commits'])} new commits...")
        ids, documents, metadatas = [], [], []
        for commit_info in report["new_commits"]:
            try:
                commit        = repo.get_commit(commit_info["sha"])
                files_changed = ", ".join([f.filename for f in commit.files[:10]])
                chunk_text    = (
                    f"Commit: {commit_info['sha']}\n"
                    f"Author: {commit.commit.author.name}\n"
                    f"Date: {str(commit.commit.author.date)}\n"
                    f"Message: {commit.commit.message.strip()}\n"
                    f"Files changed: {files_changed}"
                )
                ids.append(str(uuid.uuid4()))
                documents.append(chunk_text)
                metadatas.append({
                    "commit_sha": commit_info["sha"],
                    "author": commit.commit.author.name,
                    "date": str(commit.commit.author.date),
                    "files_changed": files_changed[:500],
                    "repo": REPO_NAME, "source": "git_commits",
                })
            except Exception as e:
                print(f"    Skipped commit {commit_info['sha']}: {e}")
        if ids:
            col_commits.add(ids=ids, documents=documents, metadatas=metadatas)
            print(f"  Added {len(ids)} new commit chunks to ChromaDB")

    # Re-index new issues only
    if report["new_issues"]:
        print(f"  Re-indexing {len(report['new_issues'])} new issues...")
        ids, documents, metadatas = [], [], []
        for issue_info in report["new_issues"]:
            try:
                issue        = repo.get_issue(issue_info["number"])
                labels       = ", ".join([l.name for l in issue.labels])
                is_pr        = issue.pull_request is not None
                comments     = list(islice(issue.get_comments(), 3))
                comment_text = "\n".join([f"Comment by {c.user.login}: {c.body[:300]}" for c in comments])
                chunk_text   = (
                    f"{'PR' if is_pr else 'Issue'} #{issue.number}: {issue.title}\n"
                    f"State: {issue.state}\nLabels: {labels}\n"
                    f"Body: {(issue.body or '')[:800]}\nComments:\n{comment_text}"
                )
                ids.append(str(uuid.uuid4()))
                documents.append(chunk_text)
                metadatas.append({
                    "issue_number": str(issue.number), "title": issue.title[:200],
                    "state": issue.state, "labels": labels, "is_pr": str(is_pr),
                    "created_at": str(issue.created_at), "repo": REPO_NAME,
                    "source": "github_issues",
                })
            except Exception as e:
                print(f"    Skipped issue #{issue_info['number']}: {e}")
        if ids:
            col_issues.add(ids=ids, documents=documents, metadatas=metadatas)
            print(f"  Added {len(ids)} new issue chunks to ChromaDB")

    # Update state with latest SHA and issue number
    latest_commits = list(islice(repo.get_commits(), 1))
    latest_issues  = list(islice(
        repo.get_issues(state='all', sort='created', direction='desc'), 1
    ))
    state["last_commit_sha"]   = latest_commits[0].sha[:8] if latest_commits else state["last_commit_sha"]
    state["last_issue_number"] = latest_issues[0].number   if latest_issues  else state["last_issue_number"]
    save_index_state(state)
    print(f"  Index state saved to {STATE_FILE}")
    print("Incremental re-indexing complete")
    return report

def save_initial_state():
    """Call this after first full index (Layer 1) to set the baseline state."""
    latest_commits = list(islice(repo.get_commits(), 1))
    latest_issues  = list(islice(
        repo.get_issues(state='all', sort='created', direction='desc'), 1
    ))
    state = {
        "last_commit_sha"  : latest_commits[0].sha[:8] if latest_commits else None,
        "last_issue_number": latest_issues[0].number   if latest_issues  else None,
        "indexed_files"    : [],
        "last_run"         : str(datetime.datetime.now()),
    }
    save_index_state(state)
    print(f"Initial index state saved: commit={state['last_commit_sha']}, issue={state['last_issue_number']}")

# Save baseline state then run staleness check
save_initial_state()
incremental_reindex()


# ============================================================
# STEP 7 — CITATION UI
# Uses the Rich library to display answers with clean formatted
# citation panels in the terminal — shows source, file, score.
# ============================================================
def format_citation(chunk, index):
    """Format a single chunk as a citation entry."""
    meta   = chunk.get("metadata", {})
    source = meta.get("source", "unknown")
    score  = chunk.get("rerank_score", chunk.get("rrf_score", "?"))

    if source == "source_code":
        label   = f"[bold cyan]SOURCE-{index}[/bold cyan] Code"
        details = f"📄 {meta.get('file_path','')} | {meta.get('type','')}: {meta.get('name','')} | Lines {meta.get('start_line','?')}-{meta.get('end_line','?')}"
    elif source == "git_commits":
        label   = f"[bold yellow]SOURCE-{index}[/bold yellow] Commit"
        details = f"🔖 {meta.get('commit_sha','')} by {meta.get('author','')} on {meta.get('date','')[:10]}"
    elif source == "github_issues":
        label   = f"[bold green]SOURCE-{index}[/bold green] Issue"
        details = f"🐛 #{meta.get('issue_number','')} — {meta.get('title','')[:60]} | State: {meta.get('state','')}"
    elif source == "readme_docs":
        label   = f"[bold magenta]SOURCE-{index}[/bold magenta] Docs"
        details = f"📖 {meta.get('file_path','')} | Chunk {meta.get('chunk_index','?')}/{meta.get('total_chunks','?')}"
    else:
        label   = f"[bold]SOURCE-{index}[/bold]"
        details = source

    preview = chunk.get("text", "")[:300].replace("\n", " ")
    return label, details, preview, score

def display_answer_with_citations(question, answer, chunks):
    """
    Display the final answer and all source citations
    using Rich for clean terminal formatting.
    """
    console.print()
    console.print(Panel(f"[bold white]{question}[/bold white]",
                        title="❓ QUESTION", border_style="bright_blue"))

    console.print(Panel(answer, title="💡 ANSWER", border_style="bright_green"))

    # Citations table
    table = Table(title="📚 SOURCE CITATIONS", box=box.ROUNDED,
                  show_header=True, header_style="bold white")
    table.add_column("#",       style="bold",         width=4)
    table.add_column("Type",    style="cyan",          width=10)
    table.add_column("Details",                        width=45)
    table.add_column("Score",   style="yellow",        width=8)
    table.add_column("Preview",                        width=40)

    for i, chunk in enumerate(chunks, 1):
        label, details, preview, score = format_citation(chunk, i)
        source_type = chunk.get("metadata", {}).get("source", "").replace("_", " ").title()
        table.add_row(str(i), source_type, details, str(score), preview[:80]+"...")

    console.print(table)
    console.print()


# ============================================================
# STEP 8 — FAITHFULNESS EVALUATOR
# Checks: is every claim in the answer backed by a retrieved chunk?
# Uses LLM to score faithfulness 0.0 to 1.0
# 1.0 = fully grounded, 0.0 = hallucinated
# ============================================================
def evaluate_faithfulness(question, answer, chunks):
    """
    Score how faithfully the answer sticks to retrieved context.
    Returns a score between 0.0 and 1.0.
    """
    context = "\n".join([c["text"][:400] for c in chunks[:5]])
    prompt  = f"""<s>[INST]
You are an evaluation assistant. Score how faithfully the ANSWER
is grounded in the CONTEXT below. A faithful answer only uses
information present in the context — it does not hallucinate.

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}

Score the faithfulness from 0.0 to 1.0 where:
1.0 = every claim is directly supported by the context
0.5 = some claims are supported, others are not
0.0 = answer is mostly hallucinated or not grounded

Respond ONLY with a JSON: {{"score": 0.0, "reason": "brief reason"}}
[/INST]"""

    response = call_llm(prompt, max_new_tokens=100)
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return {"score": float(result.get("score", 0.5)),
                    "reason": result.get("reason", "")}
    except:
        pass
    return {"score": 0.5, "reason": "Could not parse LLM evaluation"}


# ============================================================
# STEP 9 — RELEVANCE EVALUATOR
# Checks: did the retrieval actually find chunks relevant to the question?
# Uses embedding cosine similarity between question and each chunk.
# ============================================================
def evaluate_relevance(question, chunks):
    """
    Score how relevant the retrieved chunks are to the question.
    Uses cosine similarity between question embedding and chunk embeddings.
    Returns average relevance score 0.0 to 1.0.
    """
    if not chunks:
        return {"score": 0.0, "per_chunk_scores": []}

    q_embedding     = embed_model.encode([question])[0]
    chunk_texts     = [c["text"][:500] for c in chunks]
    chunk_embeddings = embed_model.encode(chunk_texts)

    # Cosine similarity
    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    per_chunk = [round(cosine_sim(q_embedding, ce), 4) for ce in chunk_embeddings]
    avg_score = round(float(np.mean(per_chunk)), 4)

    return {"score": avg_score, "per_chunk_scores": per_chunk}


# ============================================================
# STEP 10 — FULL EVALUATION PIPELINE
# Runs both evaluators on an answer and returns a combined report.
# ============================================================
def evaluate(question, answer, chunks):
    """
    Run faithfulness + relevance evaluation on a QA result.
    Returns a combined evaluation report dict.
    """
    print("  Running evaluation...")
    faithfulness = evaluate_faithfulness(question, answer, chunks)
    relevance    = evaluate_relevance(question, chunks)

    # Combined score (weighted average)
    combined = round(0.6 * faithfulness["score"] + 0.4 * relevance["score"], 4)

    report = {
        "question"       : question,
        "faithfulness"   : faithfulness,
        "relevance"      : relevance,
        "combined_score" : combined,
        "num_chunks"     : len(chunks),
        "timestamp"      : str(datetime.datetime.now()),
    }
    return report

def display_eval_report(report):
    """Display evaluation report using Rich table."""
    table = Table(title="📊 EVALUATION REPORT", box=box.ROUNDED,
                  show_header=True, header_style="bold white")
    table.add_column("Metric",   style="bold cyan",  width=20)
    table.add_column("Score",    style="bold yellow", width=10)
    table.add_column("Details",                       width=50)

    f_score = report["faithfulness"]["score"]
    r_score = report["relevance"]["score"]
    c_score = report["combined_score"]

    f_color = "green" if f_score >= 0.7 else "yellow" if f_score >= 0.4 else "red"
    r_color = "green" if r_score >= 0.7 else "yellow" if r_score >= 0.4 else "red"
    c_color = "green" if c_score >= 0.7 else "yellow" if c_score >= 0.4 else "red"

    table.add_row("Faithfulness",
                  f"[{f_color}]{f_score}[/{f_color}]",
                  report["faithfulness"].get("reason", ""))
    table.add_row("Relevance",
                  f"[{r_color}]{r_score}[/{r_color}]",
                  f"Per chunk: {report['relevance']['per_chunk_scores']}")
    table.add_row("Combined",
                  f"[{c_color}]{c_score}[/{c_color}]",
                  "0.6 x Faithfulness + 0.4 x Relevance")
    table.add_row("Chunks Used", str(report["num_chunks"]), "")

    console.print(table)


# ============================================================
# STEP 11 — FINAL COMPLETE AGENT (All Layers Combined)
# Hybrid search -> cited answer -> citation UI -> evaluation
# This is the fully polished end-to-end pipeline.
# ============================================================
def ask(question, evaluate_output=True):
    """
    The complete, polished GitHub Intelligence Agent.
    Runs hybrid search, produces a cited answer,
    displays it with Rich UI, and evaluates quality.
    """
    # Retrieve chunks via hybrid search
    chunks = hybrid_search(question, collection_names=None, top_k_final=TOP_K_FINAL)

    if not chunks:
        console.print("[red]No relevant chunks found.[/red]")
        return

    # Build context with source tags
    context = "\n\n".join([
        f"[SOURCE-{i+1}] ({c.get('metadata',{}).get('source','')}):\n{c['text'][:500]}"
        for i, c in enumerate(chunks)
    ])

    # Generate cited answer
    prompt = f"""<s>[INST]
You are a GitHub codebase intelligence agent. Answer the question using ONLY
the context provided. Cite sources using [SOURCE-N] tags.
If context is insufficient, say so clearly.

QUESTION: {question}
CONTEXT:\n{context}
[/INST]"""
    answer = call_llm(prompt, max_new_tokens=600)

    # Display with citation UI
    display_answer_with_citations(question, answer, chunks)

    # Evaluate and display report
    if evaluate_output:
        report = evaluate(question, answer, chunks)
        display_eval_report(report)

        # Append to eval report file
        all_reports = []
        if os.path.exists(EVAL_REPORT):
            with open(EVAL_REPORT) as f:
                all_reports = json.load(f)
        all_reports.append(report)
        with open(EVAL_REPORT, 'w') as f:
            json.dump(all_reports, f, indent=2)

    return answer, chunks


# ============================================================
# STEP 12 — RUN FINAL TEST QUESTIONS WITH FULL PIPELINE
# ============================================================

# Question 1
ask("How does FastAPI handle request validation?")

# Question 2
ask("What is the data flow from an API request to the response?")

# Question 3
ask("What recent bugs were reported and how were they fixed?")


# ============================================================
# STEP 13 — PRINT OVERALL EVALUATION SUMMARY
# ============================================================
if os.path.exists(EVAL_REPORT):
    with open(EVAL_REPORT) as f:
        all_reports = json.load(f)

    avg_faith   = round(sum(r["faithfulness"]["score"] for r in all_reports) / len(all_reports), 4)
    avg_rel     = round(sum(r["relevance"]["score"]    for r in all_reports) / len(all_reports), 4)
    avg_combined = round(sum(r["combined_score"]       for r in all_reports) / len(all_reports), 4)

    summary_table = Table(title="🏆 OVERALL EVALUATION SUMMARY", box=box.DOUBLE,
                          show_header=True, header_style="bold white")
    summary_table.add_column("Metric",        style="bold cyan",   width=20)
    summary_table.add_column("Average Score", style="bold yellow",  width=15)
    summary_table.add_column("Rating",                              width=15)

    def rating(score):
        if score >= 0.8: return "[green]Excellent[/green]"
        if score >= 0.6: return "[yellow]Good[/yellow]"
        if score >= 0.4: return "[orange1]Fair[/orange1]"
        return "[red]Needs Work[/red]"

    summary_table.add_row("Faithfulness",  str(avg_faith),    rating(avg_faith))
    summary_table.add_row("Relevance",     str(avg_rel),      rating(avg_rel))
    summary_table.add_row("Combined",      str(avg_combined), rating(avg_combined))
    summary_table.add_row("Total Questions", str(len(all_reports)), "")

    console.print()
    console.print(summary_table)
    console.print(f"\n[green]Full eval report saved to: {EVAL_REPORT}[/green]")

console.print()

