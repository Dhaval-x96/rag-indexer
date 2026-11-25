"""One-time script: set rag.files.is_test for files indexed before the column existed."""

from db.dbUtil import getConnection
from main import is_test_file

conn = getConnection()
try:
    with conn, conn.cursor() as cursor:
        cursor.execute("SELECT file_id, file_path FROM rag.files")
        files = cursor.fetchall()

        test_count = 0
        for file_id, file_path in files:
            is_test = is_test_file(file_path)
            cursor.execute("UPDATE rag.files SET is_test = %s WHERE file_id = %s", (is_test, file_id))
            if is_test:
                test_count += 1

        print(f"Updated {len(files)} files, {test_count} are test files")
finally:
    conn.close()
