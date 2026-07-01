# ============================================================
# STEP 1 — INSTALL DEPENDENCIES
# ============================================================
# import subprocess
# subprocess.run(["pip", "install", "-q", "chromadb", "sentence-transformers",
#                 "PyGithub", "gitpython", "tree-sitter", "tree-sitter-languages",
#                 "langchain", "langchain-community", "tiktoken"], check=True)


# ============================================================
# STEP 2 — CONFIGURATION
# Get GitHub token: github.com → Settings → Developer Settings
#                   → Personal Access Tokens → Generate (select repo scope)
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
REPO_NAME    = "momulaharinathreddy74-cyber/Object_detection-"           # <-- format: owner/repo
CHROMA_PATH  = "./chroma_db"               # where ChromaDB stores data
MAX_COMMITS  = 50                           # number of commits to pull
MAX_ISSUES   = 30                           # number of issues to pull

print(f"Config set — targeting repo: {REPO_NAME}")


# ============================================================
# STEP 3 — INITIALIZE EMBEDDING MODEL & CHROMADB
# ============================================================
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Embedding model loaded")

class MiniLMEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input):
        return model.encode(input).tolist()

embed_fn    = MiniLMEmbeddingFunction()
client      = chromadb.PersistentClient(path=CHROMA_PATH)
col_code    = client.get_or_create_collection("source_code",   embedding_function=embed_fn)
col_commits = client.get_or_create_collection("git_commits",   embedding_function=embed_fn)
col_issues  = client.get_or_create_collection("github_issues", embedding_function=embed_fn)
col_docs    = client.get_or_create_collection("readme_docs",   embedding_function=embed_fn)

print("ChromaDB initialized with 4 collections: source_code | git_commits | github_issues | readme_docs")


# ============================================================
# STEP 4 — CONNECT TO GITHUB
# ============================================================
from github import Auth, Github

if GITHUB_TOKEN:
    g = Github(auth=Auth.Token(GITHUB_TOKEN))
else:
    print("Warning: GITHUB_TOKEN is not set; using limited public GitHub access")
    g = Github()
repo = g.get_repo(REPO_NAME)
print(f"Connected to: {repo.full_name} | Stars: {repo.stargazers_count:,} | Language: {repo.language}")


# ============================================================
# STEP 5 — SOURCE CODE INGESTION
# Chunks Python files at function/class boundaries using tree-sitter
# ============================================================
import uuid
from tree_sitter_languages import get_language, get_parser

PY_LANGUAGE = get_language('python')
parser      = get_parser('python')

def extract_functions_and_classes(source_code, file_path):
    chunks = []
    try:
        tree = parser.parse(bytes(source_code, 'utf8'))
        root = tree.root_node
        for node in root.children:
            if node.type in ('function_definition', 'class_definition',
                             'decorated_definition', 'async_function_definition'):
                chunk_text = source_code[node.start_byte:node.end_byte]
                name = ""
                for child in node.children:
                    if child.type == 'identifier':
                        name = source_code[child.start_byte:child.end_byte]
                        break
                chunks.append({
                    "text": chunk_text, "name": name, "type": node.type,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1, "file_path": file_path,
                })
        if not chunks and source_code.strip():
            chunks.append({
                "text": source_code[:3000], "name": "module", "type": "module",
                "start_line": 1, "end_line": source_code.count('\n') + 1, "file_path": file_path,
            })
    except Exception as e:
        print(f"  Parse error in {file_path}: {e}")
    return chunks

def ingest_source_code(repo, collection, max_files=80):
    print("\nIngesting source code...")
    contents     = repo.get_contents("")
    all_files    = []
    total_chunks = 0
    while contents:
        item = contents.pop(0)
        if item.type == "dir":
            contents.extend(repo.get_contents(item.path))
        elif item.path.endswith(".py"):
            all_files.append(item)
        if len(all_files) >= max_files:
            break
    print(f"  Found {len(all_files)} Python files")
    for file_item in all_files:
        try:
            source = file_item.decoded_content.decode('utf-8', errors='ignore')
            chunks = extract_functions_and_classes(source, file_item.path)
            if not chunks:
                continue
            ids       = [str(uuid.uuid4()) for _ in chunks]
            documents = [c["text"][:2000] for c in chunks]
            metadatas = [{
                "file_path": c["file_path"], "name": c["name"], "type": c["type"],
                "start_line": str(c["start_line"]), "end_line": str(c["end_line"]),
                "repo": REPO_NAME, "source": "source_code",
            } for c in chunks]
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            total_chunks += len(chunks)
            print(f"  {file_item.path} -> {len(chunks)} chunks")
        except Exception as e:
            print(f"  Skipped {file_item.path}: {e}")
    print(f"Source code done — {total_chunks} chunks stored")
    return total_chunks

ingest_source_code(repo, col_code)


# ============================================================
# STEP 6 — GIT COMMITS INGESTION
# Pulls commit message + files changed + author + date
# ============================================================
def ingest_commits(repo, collection, max_commits=50):
    print(f"\nIngesting last {max_commits} commits...")
    commits   = list(repo.get_commits()[:max_commits])
    ids, documents, metadatas = [], [], []
    for commit in commits:
        try:
            sha           = commit.sha[:8]
            message       = commit.commit.message.strip()
            author        = commit.commit.author.name
            date          = str(commit.commit.author.date)
            files_changed = ", ".join([f.filename for f in commit.files[:10]])
            chunk_text    = (
                f"Commit: {sha}\nAuthor: {author}\nDate: {date}\n"
                f"Message: {message}\nFiles changed: {files_changed}"
            )
            ids.append(str(uuid.uuid4()))
            documents.append(chunk_text)
            metadatas.append({
                "commit_sha": sha, "author": author, "date": date,
                "files_changed": files_changed[:500], "repo": REPO_NAME, "source": "git_commits",
            })
        except Exception as e:
            print(f"  Skipped commit: {e}")
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Commits done — {len(ids)} commits stored")
    return len(ids)

ingest_commits(repo, col_commits, max_commits=MAX_COMMITS)


# ============================================================
# STEP 7 — GITHUB ISSUES & PRs INGESTION
# Pulls title + body + top 3 comments + labels
# ============================================================
def ingest_issues(repo, collection, max_issues=30):
    print(f"\nIngesting last {max_issues} issues & PRs...")
    from itertools import islice

    issues = list(islice(
        repo.get_issues(state="all", sort="updated"),
        max_issues
        ))
    ids, documents, metadatas = [], [], []
    for issue in issues:
        try:
            labels       = ", ".join([l.name for l in issue.labels])
            is_pr        = issue.pull_request is not None
            body         = (issue.body or "")[:1000]
            comments     = list(issue.get_comments()[:3])
            comment_text = "\n".join([f"Comment by {c.user.login}: {c.body[:300]}" for c in comments])
            chunk_text   = (
                f"{'PR' if is_pr else 'Issue'} #{issue.number}: {issue.title}\n"
                f"State: {issue.state}\nLabels: {labels}\nBody: {body}\nComments:\n{comment_text}"
            )
            ids.append(str(uuid.uuid4()))
            documents.append(chunk_text)
            metadatas.append({
                "issue_number": str(issue.number), "title": issue.title[:200],
                "state": issue.state, "labels": labels, "is_pr": str(is_pr),
                "created_at": str(issue.created_at), "repo": REPO_NAME, "source": "github_issues",
            })
        except Exception as e:
            print(f"  Skipped issue #{issue.number}: {e}")
    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Issues done — {len(ids)} issues/PRs stored")
    return len(ids)

ingest_issues(repo, col_issues, max_issues=MAX_ISSUES)


# ============================================================
# STEP 8 — README & DOCS INGESTION
# Sliding window chunking for plain text/markdown files
# ============================================================
def chunk_text(text, chunk_size=500, overlap=50):
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def ingest_docs(repo, collection):
    print("\nIngesting README & docs...")
    doc_files = []
    for readme_name in ["README.md", "README.rst", "readme.md"]:
        try:
            f = repo.get_contents(readme_name)
            doc_files.append((readme_name, f.decoded_content.decode('utf-8', errors='ignore')))
            break
        except:
            pass
    for docs_folder in ["docs", "documentation", "doc"]:
        try:
            contents = repo.get_contents(docs_folder)
            for item in contents:
                if item.path.endswith(".md") or item.path.endswith(".rst"):
                    doc_files.append((item.path, item.decoded_content.decode('utf-8', errors='ignore')))
                    if len(doc_files) >= 10:
                        break
            break
        except:
            pass
    ids, documents, metadatas = [], [], []
    for file_path, text in doc_files:
        chunks = chunk_text(text, chunk_size=300, overlap=30)
        for i, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid4()))
            documents.append(chunk)
            metadatas.append({
                "file_path": file_path, "chunk_index": str(i),
                "total_chunks": str(len(chunks)), "repo": REPO_NAME, "source": "readme_docs",
            })
        print(f"  {file_path} -> {len(chunks)} chunks")
    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Docs done — {len(ids)} chunks stored")
    return len(ids)

ingest_docs(repo, col_docs)


# ============================================================
# STEP 9 — INGESTION SUMMARY
# ============================================================
code_count = col_code.count()
commit_count = col_commits.count()
issue_count = col_issues.count()
docs_count = col_docs.count()

total = code_count + commit_count + issue_count + docs_count

print("\n" + "=" * 50)
print("       LAYER 1 INGESTION COMPLETE")
print("=" * 50)
print(f"  Repo          : {REPO_NAME}")
print(f"  Source code   : {code_count} chunks")
print(f"  Git commits   : {commit_count} chunks")
print(f"  Issues / PRs  : {issue_count} chunks")
print(f"  README / Docs : {docs_count} chunks")
print(f"  TOTAL         : {total} chunks")
print(f"  Vector DB     : {CHROMA_PATH}")
print("=" * 50)


# ============================================================
# STEP 10 — TEST QUERIES
# Verify all 4 collections are searchable
# ============================================================
def search(collection, query, n=3, label=""):
    results = collection.query(query_texts=[query], n_results=n)
    print(f"\n[{label}] Query: '{query}'")
    print("-" * 60)
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"  Result {i+1}:")
        for k, v in meta.items():
            if k not in ['repo', 'source']:
                print(f"    {k}: {v}")
        print(f"    Preview: {doc[:200]}...")
    print("-" * 60)

search(col_code,    "authentication and security",     label="SOURCE CODE")
search(col_commits, "bug fix or refactor",             label="COMMITS")
search(col_issues,  "error or exception handling",     label="ISSUES")
search(col_docs,    "how to get started installation", label="DOCS")


# ============================================================
# STEP 11 — VERIFY CHROMADB FILES ON DISK
# ============================================================
import os
print("\nChromaDB files saved on disk:")
for root, dirs, files in os.walk(CHROMA_PATH):
    for f in files:
        fpath = os.path.join(root, f)
        size  = os.path.getsize(fpath) / 1024
        print(f"  {fpath}  ({size:.1f} KB)")
print("\nLayer 1 complete! Ready for Layer 2 (BM25 + re-ranking)")
