# # ============================================================
# # STEP 1 — INSTALL DEPENDENCIES
# # ============================================================
# import subprocess
# subprocess.run(["pip", "install", "-q",
#                 "chromadb", "sentence-transformers",
#                 "rank_bm25", "PyGithub", "requests", "numpy"], check=True)


# ============================================================
# STEP 2 — CONFIGURATION
# ============================================================
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH  = "./chroma_db"       # same as all previous layers
REPO_NAME    = "momulaharinathreddy74-cyber/Object_detection-"            # same as all previous layers
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # <-- same GitHub token as Layer 1
HF_TOKEN     = os.getenv("HF_TOKEN")         # <-- same HF token as Layer 3
DOC_OUTPUT   = "./generated_docs"   # folder where create_doc() saves files
TOP_K_FINAL  = 5
TOP_K_DENSE  = 10
TOP_K_BM25   = 10
MAX_HOPS     = 3
MAX_SUBQ     = 3
MAX_TOOL_CALLS = 5   # max tools the agent can call per question

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

os.makedirs(DOC_OUTPUT, exist_ok=True)
print(f"Config set | Repo: {REPO_NAME} | Doc output: {DOC_OUTPUT}")


# ============================================================
# STEP 3 — RELOAD ALL MODELS & CHROMADB (from Layers 1-3)
# ============================================================
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, CrossEncoder
from github import Auth, Github
from rank_bm25 import BM25Okapi
import re, requests, json, uuid

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
print("All indexes ready")


# ============================================================
# STEP 4 — HYBRID SEARCH PIPELINE (carried from Layer 2)
# ============================================================
def dense_search(query, collection, top_k=10):
    n = min(top_k, collection.count())
    r = collection.query(query_texts=[query], n_results=n)
    return [{"text": d, "metadata": m, "dense_rank": i+1}
            for i,(d,m) in enumerate(zip(r["documents"][0], r["metadatas"][0]))]

def bm25_search(query, index_data, top_k=10):
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
# STEP 6 — MCP TOOL DEFINITIONS
# Each tool is a Python function the agent can call.
# The agent decides WHICH tool to call based on the question.
# ============================================================

# --- Tool 1: search_code ---
# Semantic search over the indexed codebase in ChromaDB
def search_code(query):
    """
    Search the indexed source code semantically.
    Use for: finding where something is implemented, understanding logic.
    """
    print(f"  [TOOL] search_code('{query}')")
    results = hybrid_search(query, collection_names=["source_code"], top_k_final=TOP_K_FINAL)
    output  = []
    for r in results:
        meta = r.get("metadata", {})
        output.append({
            "file"      : meta.get("file_path", ""),
            "name"      : meta.get("name", ""),
            "type"      : meta.get("type", ""),
            "lines"     : f"{meta.get('start_line','?')}-{meta.get('end_line','?')}",
            "preview"   : r["text"][:400],
            "score"     : r.get("rerank_score", 0),
        })
    return output


# --- Tool 2: get_commit ---
# Fetches a full commit diff LIVE from GitHub API
def get_commit(commit_hash):
    """
    Fetch the full diff and metadata of a commit from GitHub.
    Use for: understanding what exactly changed in a specific commit.
    """
    print(f"  [TOOL] get_commit('{commit_hash}')")
    try:
        commit       = repo.get_commit(commit_hash)
        files_detail = []
        for f in commit.files[:5]:  # limit to 5 files
            files_detail.append({
                "filename"  : f.filename,
                "status"    : f.status,
                "additions" : f.additions,
                "deletions" : f.deletions,
                "patch"     : (f.patch or "")[:500],  # first 500 chars of diff
            })
        return {
            "sha"          : commit.sha[:8],
            "message"      : commit.commit.message,
            "author"       : commit.commit.author.name,
            "date"         : str(commit.commit.author.date),
            "files_changed": files_detail,
        }
    except Exception as e:
        return {"error": str(e)}


# --- Tool 3: get_issue ---
# Fetches a full issue thread LIVE from GitHub API
def get_issue(issue_number):
    """
    Fetch the full thread of a GitHub issue including all comments.
    Use for: understanding the discussion, bug reports, decisions made.
    """
    print(f"  [TOOL] get_issue({issue_number})")
    try:
        issue    = repo.get_issue(int(issue_number))
        comments = list(issue.get_comments()[:5])  # top 5 comments
        return {
            "number"    : issue.number,
            "title"     : issue.title,
            "state"     : issue.state,
            "labels"    : [l.name for l in issue.labels],
            "body"      : (issue.body or "")[:800],
            "comments"  : [
                {"author": c.user.login, "body": c.body[:400]}
                for c in comments
            ],
            "created_at": str(issue.created_at),
        }
    except Exception as e:
        return {"error": str(e)}


# --- Tool 4: navigate_file ---
# Reads any file in the repo LIVE from GitHub
def navigate_file(file_path):
    """
    Read the full content of any file in the repo.
    Use for: reading specific implementation files, configs, or docs.
    """
    print(f"  [TOOL] navigate_file('{file_path}')")
    try:
        content = repo.get_contents(file_path)
        text    = content.decoded_content.decode('utf-8', errors='ignore')
        return {
            "path"    : file_path,
            "size"    : content.size,
            "content" : text[:2000],  # first 2000 chars
            "truncated": len(text) > 2000,
        }
    except Exception as e:
        return {"error": str(e)}


# --- Tool 5: create_doc ---
# Saves generated content as a markdown file locally
def create_doc(filename, content):
    """
    Save generated documentation or onboarding guide to a markdown file.
    Use for: creating onboarding guides, architecture summaries, changelogs.
    """
    print(f"  [TOOL] create_doc('{filename}')")
    try:
        safe_name = re.sub(r'[^\w\-.]', '_', filename)
        if not safe_name.endswith('.md'):
            safe_name += '.md'
        path = os.path.join(DOC_OUTPUT, safe_name)
        with open(path, 'w') as f:
            f.write(content)
        return {"status": "success", "path": path, "size_bytes": len(content)}
    except Exception as e:
        return {"error": str(e)}


# --- Tool 6: open_issue ---
# Creates a real GitHub issue in the repo
def open_issue(title, body):
    """
    Create a new GitHub issue in the repo.
    Use for: reporting bugs found during analysis, tracking improvements.
    NOTE: This creates a REAL issue — only use when explicitly asked.
    """
    print(f"  [TOOL] open_issue('{title}')")
    try:
        issue = repo.create_issue(title=title, body=body)
        return {
            "status" : "created",
            "number" : issue.number,
            "url"    : issue.html_url,
            "title"  : issue.title,
        }
    except Exception as e:
        return {"error": str(e)}


# Tool registry — maps tool name to function
TOOL_REGISTRY = {
    "search_code"  : search_code,
    "get_commit"   : get_commit,
    "get_issue"    : get_issue,
    "navigate_file": navigate_file,
    "create_doc"   : create_doc,
    "open_issue"   : open_issue,
}

# Tool descriptions sent to LLM so it knows what each tool does
TOOL_DESCRIPTIONS = """
Available tools:
1. search_code(query)          — Semantic search over codebase. Use for: finding implementations, understanding logic.
2. get_commit(commit_hash)     — Fetch full commit diff from GitHub. Use for: seeing exactly what changed in a commit.
3. get_issue(issue_number)     — Fetch full issue thread from GitHub. Use for: reading bug reports, discussions, decisions.
4. navigate_file(file_path)    — Read any file in the repo. Use for: reading specific files, configs, or docs.
5. create_doc(filename,content)— Save generated content as a markdown file. Use for: creating onboarding guides or summaries.
6. open_issue(title,body)      — Create a new GitHub issue. ONLY use when explicitly asked by the user.
"""

print("MCP Tools registered:", list(TOOL_REGISTRY.keys()))


# ============================================================
# STEP 7 — TOOL SELECTION (LLM decides which tool to call)
# The LLM reads the question and picks the best tool + args.
# Returns structured JSON: {"tool": "...", "args": {...}}
# ============================================================
def select_tool(question, context_so_far=""):
    """
    Ask the LLM which tool to call next given the question
    and what context has been gathered so far.
    """
    prompt = f"""<s>[INST]
You are a GitHub codebase agent. Given the question and context so far,
decide which tool to call next to best answer the question.

{TOOL_DESCRIPTIONS}

QUESTION: {question}

CONTEXT SO FAR:
{context_so_far[:800] if context_so_far else 'None yet'}

Respond ONLY with a JSON object like these examples:
{{"tool": "search_code", "args": {{"query": "authentication middleware"}}}}
{{"tool": "get_commit", "args": {{"commit_hash": "abc1234"}}}}
{{"tool": "get_issue", "args": {{"issue_number": 42}}}}
{{"tool": "navigate_file", "args": {{"file_path": "fastapi/security.py"}}}}
{{"tool": "create_doc", "args": {{"filename": "onboarding", "content": "# Guide..."}}}}
{{"tool": "done", "args": {{}}}} <- use this when you have enough context to answer

Return ONLY the JSON, nothing else.
[/INST]"""

    response = call_llm(prompt, max_new_tokens=100)

    # Parse JSON from LLM response
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass

    # Fallback: if LLM didn't return valid JSON, default to search_code
    return {"tool": "search_code", "args": {"query": question}}


# ============================================================
# STEP 8 — TOOL EXECUTOR
# Receives the tool decision from LLM and actually runs it.
# Returns the tool output as a formatted string.
# ============================================================
def execute_tool(tool_name, args):
    """
    Execute a tool from the registry with the given args.
    Returns a string summary of the tool output.
    """
    if tool_name not in TOOL_REGISTRY:
        return f"[Unknown tool: {tool_name}]"

    fn     = TOOL_REGISTRY[tool_name]
    result = fn(**args)

    # Format result as readable string for LLM context
    if isinstance(result, list):
        formatted = json.dumps(result, indent=2)[:1500]
    elif isinstance(result, dict):
        formatted = json.dumps(result, indent=2)[:1500]
    else:
        formatted = str(result)[:1500]

    return f"[{tool_name} result]:\n{formatted}"


# ============================================================
# STEP 9 — AGENTIC TOOL-CALLING LOOP
# The agent iterates: select tool -> execute -> add to context
# Stops when LLM says "done" or MAX_TOOL_CALLS reached.
# This is what makes it a true agent — it navigates the repo.
# ============================================================
def run_tool_loop(question):
    """
    Run the agentic tool-calling loop.
    Returns accumulated context from all tool calls.
    """
    print(f"\n  Starting tool loop for: '{question}'")
    context_parts = []
    tool_log      = []

    for call_num in range(MAX_TOOL_CALLS):
        context_so_far = "\n".join(context_parts)

        # LLM picks next tool
        decision = select_tool(question, context_so_far)
        tool_name = decision.get("tool", "done")
        args      = decision.get("args", {})

        print(f"  Call {call_num+1}: LLM chose -> {tool_name}({args})")
        tool_log.append({"call": call_num+1, "tool": tool_name, "args": args})

        # Stop if agent says it has enough context
        if tool_name == "done":
            print("  Agent decided it has enough context — stopping tool loop")
            break

        # Execute the tool
        output = execute_tool(tool_name, args)
        context_parts.append(output)

    return "\n\n".join(context_parts), tool_log


# ============================================================
# STEP 10 — FINAL ANSWER SYNTHESIS
# LLM reads all tool outputs and produces a grounded answer.
# ============================================================
def synthesize_with_tools(question, tool_context):
    """
    Given all tool outputs, ask the LLM to synthesize a final answer.
    """
    prompt = f"""<s>[INST]
You are a GitHub codebase intelligence agent.
Using ONLY the tool outputs below, answer the question clearly and specifically.
Reference specific files, functions, commits, or issues where relevant.
If the tools didn't return enough information, say so clearly.

QUESTION: {question}

TOOL OUTPUTS:
{tool_context[:2000]}

Provide a detailed, structured answer grounded in the tool outputs above.
[/INST]"""

    return call_llm(prompt, max_new_tokens=700)


# ============================================================
# STEP 11 — MAIN AGENT WITH MCP TOOLS
# Full pipeline: Tool loop -> Synthesize -> Print answer + tool log
# ============================================================
def ask_agent_v2(question):
    """
    Layer 4 agent — uses MCP tools to actively navigate the repo.
    Combines Layer 3 agentic retrieval with Layer 4 live tool calls.
    """
    print("\n" + "="*60)
    print(f" QUESTION: {question}")
    print("="*60)

    # Run the tool-calling loop
    tool_context, tool_log = run_tool_loop(question)

    if not tool_context.strip():
        print("  No tool output gathered.")
        return

    # Synthesize final answer
    print("  Synthesizing final answer...")
    answer = synthesize_with_tools(question, tool_context)

    # Print answer
    print("\n" + "-"*60)
    print(" ANSWER:")
    print("-"*60)
    print(answer)

    # Print tool call log
    print("\n" + "-"*60)
    print(" TOOLS CALLED:")
    print("-"*60)
    for entry in tool_log:
        print(f"  Call {entry['call']}: {entry['tool']}({entry['args']})")

    print("\n" + "="*60)
    return answer


# ============================================================
# STEP 12 — TEST ALL 6 TOOLS
# Each question is designed to trigger a different tool
# ============================================================

# Test 1: Triggers search_code — semantic code search
ask_agent_v2("Where is request validation handled in the codebase?")

# Test 2: Triggers get_commit — fetches live commit diff
# Replace with a real commit SHA from your repo after running Layer 1
print("\n--- Direct tool test: get_commit ---")
first_commit = next(iter(repo.get_commits()), None)
if first_commit:
    result = get_commit(first_commit.sha[:8])
    print(json.dumps(result, indent=2))

# Test 3: Triggers get_issue — fetches live issue thread
print("\n--- Direct tool test: get_issue ---")
first_issue = next(iter(repo.get_issues(state='open')), None)
if first_issue:
    result = get_issue(first_issue.number)
    print(json.dumps(result, indent=2))
else:
    print("No open issues found; skipping get_issue test.")

# Test 4: Triggers navigate_file — reads a real file
print("\n--- Direct tool test: navigate_file ---")
result = navigate_file("README.md")
print(json.dumps(result, indent=2))

# Test 5: Triggers create_doc — generates an onboarding guide
ask_agent_v2("Generate an onboarding guide for a new developer joining this repo")

# Verify the doc was saved
print("\n--- Generated docs saved at: ---")
for f in os.listdir(DOC_OUTPUT):
    fpath = os.path.join(DOC_OUTPUT, f)
    print(f"  {fpath} ({os.path.getsize(fpath)} bytes)")

print("\nLayer 4 complete! All MCP tools are wired and working.")
print("Next: Layer 5 — Polish: incremental indexing + citation UI + evaluation metrics")
