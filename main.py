from fastapi import FastAPI, BackgroundTasks
from db.dbUtil import getConnection
import os
import tempfile
from git import Repo
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import shutil

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



app = FastAPI()

def index_repo(repo_id: str):
    print(f"Indexing repository {repo_id}...")
    conn = getConnection()
    
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
                "INSERT INTO rag.chunks (file_id, chunk_text, chunk_index, start_line, end_line) VALUES %s" \
                "RETURNING chunk_id, chunk_text",
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

    




@app.post("/index/{repo_id}", status_code=202)
def start_index(repo_id:str, background_tasks: BackgroundTasks):
    background_tasks.add_task(index_repo, repo_id)
    return {"status": "accepted", "repo_id": repo_id}

