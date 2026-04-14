from psycopg_pool import ConnectionPool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Construct the database URL from environment variables
DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

# Initialize the connection pool
pool = ConnectionPool(DB_URL)

def execute_sql_file(file_path):
    """Execute SQL commands from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        sql = f.read()

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)

if __name__ == "__main__":
    sql_file = "src/schemas/internal_database_seed.sql"
    print(f"Executing {sql_file}...")
    execute_sql_file(sql_file)
    print(f"Finished executing {sql_file}.")