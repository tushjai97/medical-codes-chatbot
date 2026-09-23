"""
Database Setup Script
Creates tables and indices for medical coding system
"""

import asyncio
import asyncpg
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '1024'))

CREATE_EXTENSIONS = """
CREATE EXTENSION IF NOT EXISTS vector;
"""

CREATE_CPT_TABLE = f"""
CREATE TABLE IF NOT EXISTS cpt_codes (
    id SERIAL PRIMARY KEY,
    cpt_code VARCHAR(5) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(50),
    code_status VARCHAR(20),

    -- Vector embedding
    embedding vector({EMBEDDING_DIM}),

    -- Full-text search (auto-generated)
    description_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', description || ' ' || COALESCE(category, ''))
    ) STORED,

    -- Metadata
    usage_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

CREATE_ICD10_TABLE = f"""
CREATE TABLE IF NOT EXISTS icd10_codes (
    id SERIAL PRIMARY KEY,
    icd10_code VARCHAR(10) UNIQUE NOT NULL,
    description TEXT NOT NULL,

    -- Hierarchy (extracted from code pattern)
    chapter VARCHAR(10),
    block VARCHAR(20),

    -- Vector embedding
    embedding vector({EMBEDDING_DIM}),

    -- Full-text search
    description_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', description)
    ) STORED,

    -- Metadata
    usage_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

# Non-vector indices are safe to build before data is loaded.
CREATE_NON_VECTOR_INDICES = """
CREATE INDEX IF NOT EXISTS idx_cpt_fts
    ON cpt_codes USING GIN (description_tsv);

CREATE INDEX IF NOT EXISTS idx_cpt_category
    ON cpt_codes (category);

CREATE INDEX IF NOT EXISTS idx_icd10_fts
    ON icd10_codes USING GIN (description_tsv);

CREATE INDEX IF NOT EXISTS idx_icd10_chapter
    ON icd10_codes (chapter);
"""

# ivfflat indices cluster centroids from the data present at CREATE INDEX
# time. Building them on an empty table produces near-meaningless clusters
# and badly degrades nearest-neighbor recall, so these must be (re)built
# AFTER the loader scripts have inserted data -- see rebuild_vector_indices().
CREATE_VECTOR_INDICES = """
CREATE INDEX IF NOT EXISTS idx_cpt_vector
    ON cpt_codes USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_icd10_vector
    ON icd10_codes USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


async def rebuild_cpt_vector_index(conn):
    """(Re)build idx_cpt_vector from current table contents.

    Must be called after cpt_codes is populated -- an ivfflat index built
    on an empty table has degenerate clusters and will return poor/incorrect
    nearest-neighbor results even once data is inserted later, unless the
    index is rebuilt.
    """
    await conn.execute("DROP INDEX IF EXISTS idx_cpt_vector;")
    await conn.execute("""
        CREATE INDEX idx_cpt_vector
            ON cpt_codes USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
    """)


async def rebuild_icd10_vector_index(conn):
    """(Re)build idx_icd10_vector from current table contents. See
    rebuild_cpt_vector_index() for why this must run after data is loaded."""
    await conn.execute("DROP INDEX IF EXISTS idx_icd10_vector;")
    await conn.execute("""
        CREATE INDEX idx_icd10_vector
            ON icd10_codes USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
    """)


async def setup_database(reset: bool = False):
    """Setup database schema

    Args:
        reset: If True, drop existing cpt_codes/icd10_codes tables first.
            Required when the embedding dimension changes (e.g. switching
            embedding models), since existing embeddings become invalid.
    """
    db_url = os.getenv('NEON_DATABASE_URL')
    if not db_url:
        raise ValueError("NEON_DATABASE_URL environment variable not set")

    print("Connecting to database...")
    conn = await asyncpg.connect(db_url)

    try:
        print("Creating pgvector extension...")
        await conn.execute(CREATE_EXTENSIONS)

        if reset:
            print(f"Resetting tables for embedding dimension {EMBEDDING_DIM}...")
            await conn.execute("DROP TABLE IF EXISTS cpt_codes;")
            await conn.execute("DROP TABLE IF EXISTS icd10_codes;")

        print("Creating cpt_codes table...")
        await conn.execute(CREATE_CPT_TABLE)

        print("Creating icd10_codes table...")
        await conn.execute(CREATE_ICD10_TABLE)

        print("Creating non-vector indices...")
        await conn.execute(CREATE_NON_VECTOR_INDICES)

        print("Note: vector indices are NOT created here -- run")
        print("      rebuild_vector_indices() (or the loader scripts,")
        print("      which do this automatically) after loading data.")

        print("Database setup complete!")

        # Verify
        cpt_count = await conn.fetchval("SELECT COUNT(*) FROM cpt_codes")
        icd10_count = await conn.fetchval("SELECT COUNT(*) FROM icd10_codes")

        print(f"\nCurrent data:")
        print(f"   CPT codes: {cpt_count}")
        print(f"   ICD-10 codes: {icd10_count}")

    finally:
        await conn.close()


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    asyncio.run(setup_database(reset=reset))
