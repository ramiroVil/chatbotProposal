import os
import uuid
import psycopg
from pgvector.psycopg import register_vector

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://root:Admin456@localhost:5432/db_013025")

def get_conn():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)  # habilita enviar/leer el tipo vector
    return conn

def new_uuid():
    return str(uuid.uuid4())