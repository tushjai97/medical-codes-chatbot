"""
Load CPT codes from CSV and generate embeddings
"""

import asyncio
import asyncpg
import csv
import os
import sys
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.embeddings import get_embedding_service
from app.config import settings
from setup_database import rebuild_cpt_vector_index

load_dotenv()

CPT_FILE = Path(__file__).parent.parent.parent / "data" / "all-2025-cpt-codes.csv"
BATCH_SIZE = 100


async def load_cpt_codes():
    """Load CPT codes with embeddings"""

    # Load CSV
    print(f"Reading CPT codes from {CPT_FILE}...")
    cpt_data = []

    with open(CPT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpt_data.append({
                'category': row['Procedure Code Category'],
                'code': row['CPT Codes'].strip(),
                'description': row['Procedure Code Descriptions'].strip(),
                'status': row['Code Status']
            })

    print(f"Loaded {len(cpt_data)} CPT codes")

    # Generate embeddings
    print(f"\nGenerating embeddings via {settings.VOYAGE_MODEL_NAME}...")
    embedding_service = get_embedding_service(
        api_key=settings.VOYAGE_API_KEY,
        model_name=settings.VOYAGE_MODEL_NAME,
        dimension=settings.EMBEDDING_DIM,
    )

    descriptions = [item['description'] for item in cpt_data]
    try:
        embeddings = embedding_service.generate_embeddings_batch(descriptions, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Failed to generate embeddings: {e}")
        raise

    # Add embeddings to data
    for i, item in enumerate(cpt_data):
        item['embedding'] = embeddings[i].tolist()

    # Insert into database
    print("\nInserting into database...")
    db_url = os.getenv('NEON_DATABASE_URL')
    conn = await asyncpg.connect(db_url)

    try:
        # Clear existing data
        await conn.execute("TRUNCATE TABLE cpt_codes CASCADE")

        # Batch insert
        insert_query = """
            INSERT INTO cpt_codes (cpt_code, description, category, code_status, embedding)
            VALUES ($1, $2, $3, $4, $5)
        """

        for item in tqdm(cpt_data, desc="Inserting CPT codes"):
            # Convert embedding list to pgvector format
            embedding_str = '[' + ','.join(map(str, item['embedding'])) + ']'
            await conn.execute(
                insert_query,
                item['code'],
                item['description'],
                item['category'],
                item['status'],
                embedding_str
            )

        # Rebuild the vector index now that data exists (an ivfflat index
        # built on an empty table has degenerate clusters)
        print("\nRebuilding vector index...")
        await rebuild_cpt_vector_index(conn)

        # Verify
        count = await conn.fetchval("SELECT COUNT(*) FROM cpt_codes")
        print(f"\nSuccessfully loaded {count} CPT codes")

        # Sample query
        sample = await conn.fetchrow("""
            SELECT cpt_code, description, category
            FROM cpt_codes
            LIMIT 1
        """)
        print(f"\nSample record:")
        print(f"   Code: {sample['cpt_code']}")
        print(f"   Category: {sample['category']}")
        print(f"   Description: {sample['description'][:80]}...")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(load_cpt_codes())
