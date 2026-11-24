from fastapi import FastAPI, BackgroundTasks
from db.dbUtil import getConnection
import os
import tempfile
from git import Repo
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import shutil
from pydantic import BaseModel
import requests

model = SentenceTransformer("all-MiniLM-L6-v2")


ALLOWED_EXTENSIONS = {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt",
        ".cpp", ".c", ".h", ".cs", ".rb", ".go", ".rs", ".php",
        ".html", ".css", ".sql", ".md",
    }
SKIP_DIRS = {
        ".git", "node_modules", "target", "build", "dist", "out",
        "venv", ".venv", "__pycache__", ".idea", ".vscode",
        "coverage", ".next", "vendor",".github",
    }
SKIP_SUFFIXES = (".min.js", ".min.css")
CHUNK_SIZE=50
OVERLAP=10
MAX_DISTANCE = 0.6


app = FastAPI()


class AskRequest(BaseModel):
    question: str



@app.post("/index/{repo_id}", status_code=202)
def start_index(repo_id:str, background_tasks: BackgroundTasks):
    background_tasks.add_task(index_repo, repo_id)
    return {"status": "accepted", "repo_id": repo_id}


@app.post("/repo/{repo_id}/ask")
def ask_question(repo_id: str, request: AskRequest):
    snippets = search_chunk(repo_id, request)
    
    ## USE FOR COST OPTIMIZATION
    ## ADD THREADSOLD FOR FETCHING ACCURATE  
    if not snippets or snippets[0]["distance"] > MAX_DISTANCE:
        return {
            "question":request.question,
            "answer":"I don't know. No relevant code found in this repo.",
            "snippets":[]
        }
    
    answer = generate_answer(request.question, snippets)
    return {"question": request.question, "answer": answer, "snippets": snippets}


def search_chunk(repo_id: str, request: AskRequest):
    vector = model.encode(request.question).tolist()
    conn = getConnection()
    try:
        with conn, conn.cursor() as cursor:
            cursor.execute("SELECT f.file_path, c.start_line, c.end_line, " \
            " c.chunk_text, e.embedding <=> %s::vector as distance FROM rag.embedding e JOIN rag.chunks c " \
            " ON  e.chunk_id = c.chunk_id JOIN rag.files f ON f.file_id = c.file_id " \
            " WHERE f.repo_id = %s order by distance limit 30", (vector, repo_id))
            rows = cursor.fetchall()
    finally:
        conn.close()

    # Keep only the best chunk per file, so one file can't fill all 5 slots.
    snippets = []
    seen = set()
    for p, s, e, text, d in rows:
        if p in seen:
            continue
        seen.add(p)
        snippets.append({"file_path": p,
                         "start_line": s,
                         "end_line": e,
                         "code": text,
                         "distance": d})
        if len(snippets) == 5:
            break
    return snippets



def key_search(repo_id,question, limit=20):
    conn = getConnection()
    # Steps:
    # 1. plainto_tsquery: split question into words -> 'bash' & 'sandbox'
    # 2. replace & with |: any word can match -> 'bash' | 'sandbox'
    # 3. @@: keep chunks that contain at least one word (uses GIN index)
    # 4. ts_rank: score chunks, more matching words = higher score
    # 5. JOIN files: get file_path for each chunk conn = getConnection()


    try:
        with conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT c.chunk_id, f.file_path, c.chunk_text,
                       c.start_line, c.end_line,
                       ts_rank(c.text_search, q) AS score
                FROM rag.chunks c
                JOIN rag.files f ON f.file_id = c.file_id,
                     to_tsquery('simple', replace(plainto_tsquery('simple', %s)::text, '&', '|')) q
                WHERE f.repo_id = %s
                  AND c.text_search @@ q
                ORDER BY score DESC
                LIMIT %s
            """, (question, repo_id, limit))
            return cursor.fetchall()
    finally:
        conn.close()

def generate_answer(question, snippets):
    context = "\n\n".join(
               f"[{i}] {s['file_path']} (lines {s['start_line']}-{s['end_line']})\n{s['code']}"
               for i, s in enumerate(snippets, start=1)
    )

    token = os.getenv("ANTHROPIC_API_KEY")
    response = requests.post( "https://api.anthropic.com/v1/messages",
                  headers={
                        "Content-Type": "application/json",
                        "x-api-key": token,
                        "anthropic-version": "2026-10-21",
                    },
                   json={
                       "model": "claude-haiku-5-5",
                        "max_tokens": 2048,
                        "system":
                            "You explain codebases. Answer only from the code snippets given. "
                            "Cite the snippets you used like [1] or [2]. "
                            "If the snippets don't contain the answer, say you don't know. "
                            "Match the length to the question: a few sentences for simple "
                            "questions, more detail with steps for complex ones. "
                            "Don't pad the answer.",
                        "messages": [
                            {"role": "user", "content": f"Code snippets:\n\n{context}\n\nQuestion: {question}"}
                        ],
                   },timeout=60)
    response.raise_for_status()

    data = response.json()
    answer = "".join(b["text"] for b in data["content"] if b["type"] == "text")
    if data.get("stop_reason") == "max_tokens":
        answer += "\n\n[Answer was cut off. Ask a narrower question for the rest.]"
    return answer
    

def index_repo(repo_id: str):
    print(f"Indexing repository {repo_id}...")
    conn = getConnection()
    temp_dir = None
    
    try:
         ## tX1
        with conn, conn.cursor() as cursor:
            cursor.execute("UPDATE rag.repo SET status = 'INDEXING' WHERE repo_id = %s RETURNING git_url", (repo_id,))
            # conn.commit() ## not required
            row = cursor.fetchone()
            if row is None:
                print(f"Repository {repo_id} has no git_url. Skipping indexing.")
                return
            else:
                git_url = row[0]
            
            print(f"Repository {repo_id} git_url: {git_url}")
            temp_dir = tempfile.mkdtemp()
            print(f"Cloning repository {repo_id} into temporary directory...")
            Repo.clone_from(git_url, temp_dir, depth=1)
            print(f"Repository {repo_id} cloned into successfully")
            paths = find_file_path(temp_dir)

            ## tX2
        with conn, conn.cursor() as cursor:
            file_rows = save_files(cursor, repo_id, paths)
            print(f"Saved {len(paths)} files for repo {repo_id}")
            chunk_rows = save_chunks(cursor, temp_dir, file_rows)
            total = save_embeddings(cursor, chunk_rows)
            print(f"Saved {len(file_rows)} files and {total} chunks")

            cursor.execute(
                "UPDATE rag.repo SET status = 'DONE' WHERE repo_id = %s",
                (repo_id,),
            )
        print(f"Saved {len(file_rows)} files, {len(chunk_rows)} chunks "
              f"and {total} embeddings")

    except Exception as e:
        print(f"Indexing failed for {repo_id}: {e}")
        with conn, conn.cursor() as cursor:
            cursor.execute(
                "UPDATE rag.repo SET status = 'FAILED' WHERE repo_id = %s",
                (repo_id,),
            )
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        conn.close()   

def save_embeddings(cursor,chunk_row):
    chunk_ids = [row[0] for row in chunk_row]
    text = [row[1] for row in chunk_row]
    vectors = model.encode(text, batch_size=64, show_progress_bar=True)

    execute_values(
        cursor,
        "INSERT INTO rag.embedding (chunk_id, embedding) VALUES %s",
        [(cid, vec.tolist()) for cid, vec in zip(chunk_ids, vectors)],
        template="(%s, %s::vector)",
        page_size=500,
    )
    return len(chunk_ids)



def save_chunks(cursor, temp_dir, file_rows):
    chunk_rows=[]
    for file_id, file_path in file_rows:
        full_path = os.path.join(temp_dir, file_path)
        for index, (text, start, end) in enumerate(chunk_file(full_path)):
            chunk_rows.append((file_id, text, index, start, end))

    rows = execute_values(
                cursor,
                "INSERT INTO rag.chunks (file_id, chunk_text, chunk_index, start_line, end_line) VALUES %s " \
                " RETURNING chunk_id, chunk_text",
                chunk_rows,
                page_size=500,
                fetch=True
            )
    return rows

def chunk_file(full_path):
    with open(full_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # A NUL byte means binary or broken text; Postgres can't store it.
    if "\x00" in content:
        print(f"Skipping file with NUL bytes: {full_path}")
        return []

    lines = content.splitlines(keepends=True)
    chunks = []
    step = CHUNK_SIZE - OVERLAP
    for start in range(0, len(lines), step):
        end = min(start + CHUNK_SIZE, len(lines))
        text = "".join(lines[start:end]).strip()
        if text:
            chunks.append((text, start + 1, end))
        if end == len(lines):
            break
    return chunks


def find_file_path(temp_dir):
    paths = []
    for root, dirs, files in os.walk(temp_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in ALLOWED_EXTENSIONS and not file.endswith(SKIP_SUFFIXES):
                rel_path = os.path.relpath(os.path.join(root, file), temp_dir)
                paths.append((rel_path, ext.lstrip(".")))
    return paths


def save_files(cursor, repo_id, paths):
    return execute_values(
            cursor,
            "INSERT INTO rag.files (repo_id, file_path, extension) VALUES %s RETURNING file_id, file_path",
            [(repo_id, path,ext) for path, ext in paths],
            fetch=True
        )

