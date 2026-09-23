"""
Load ICD-10 codes from text file and generate embeddings
Extracts hierarchy information from code patterns
"""

import asyncio
import asyncpg
import os
import sys
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.embeddings import get_embedding_service
from app.config import settings
from setup_database import rebuild_icd10_vector_index

load_dotenv()

ICD10_FILE = Path(__file__).parent.parent.parent / "data" / "icd10cm-codes-2025.txt"
BATCH_SIZE = 500  # Voyage allows up to 1000 texts / 1M tokens per request


def parse_icd10_line(line: str) -> dict:
    """
    Parse ICD-10 line: "CODE    DESCRIPTION"

    Returns:
        dict with code, description, chapter, block
    """
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        return None

    code = parts[0].strip()
    description = parts[1].strip()

    # Extract hierarchy from code pattern
    chapter = None
    block = None

    if len(code) >= 1:
        # Chapter: first letter
        letter = code[0]
        if letter.isalpha():
            chapter = f"{letter}00-{letter}99"

    if len(code) >= 3:
        # Block: category level
        if code[1:3].isdigit():
            start = int(code[1:3]) // 10 * 10
            end = start + 9
            block = f"{code[0]}{start:02d}-{code[0]}{end:02d}"

    return {
        'code': code,
        'description': description,
        'chapter': chapter,
        'block': block
    }


async def load_icd10_codes(limit: int = None):
    """Load ICD-10 codes with embeddings

    Args:
        limit: If set, only load/embed the first N codes (useful for a
            quick validation run before committing to the full ~74k set)
    """

    # Parse file
    print(f"Reading ICD-10 codes from {ICD10_FILE}...")
    icd10_data = []

    with open(ICD10_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parsed = parse_icd10_line(line)
                if parsed:
                    icd10_data.append(parsed)

    if limit:
        icd10_data = icd10_data[:limit]

    print(f"Loaded {len(icd10_data)} ICD-10 codes")

    # Generate embeddings
    print(f"\nGenerating embeddings via {settings.VOYAGE_MODEL_NAME} (this will take a few minutes)...")
    embedding_service = get_embedding_service(
        api_key=settings.VOYAGE_API_KEY,
        model_name=settings.VOYAGE_MODEL_NAME,
        dimension=settings.EMBEDDING_DIM,
    )

    descriptions = [item['description'] for item in icd10_data]

    # Process in batches (one Voyage API call per batch)
    all_embeddings = []
    for i in tqdm(range(0, len(descriptions), BATCH_SIZE), desc="Embedding batches"):
        batch = descriptions[i:i + BATCH_SIZE]
        try:
            batch_embeddings = embedding_service.generate_embeddings_batch(batch, batch_size=BATCH_SIZE)
        except Exception as e:
            print(f"Failed to embed batch starting at index {i}: {e}")
            raise
        all_embeddings.extend(batch_embeddings.tolist())

    # Add embeddings to data
    for i, item in enumerate(icd10_data):
        item['embedding'] = all_embeddings[i]

    # Insert into database
    print("\nInserting into database...")
    db_url = os.getenv('NEON_DATABASE_URL')
    conn = await asyncpg.connect(db_url)

    try:
        # Clear existing data
        await conn.execute("TRUNCATE TABLE icd10_codes CASCADE")

        # Batch insert using executemany
        insert_query = """
            INSERT INTO icd10_codes (icd10_code, description, chapter, block, embedding)
            VALUES ($1, $2, $3, $4, $5)
        """

        # Prepare data with embeddings converted to pgvector format
        records = [
            (
                item['code'],
                item['description'],
                item['chapter'],
                item['block'],
                '[' + ','.join(map(str, item['embedding'])) + ']'  # Convert list to pgvector string
            )
            for item in icd10_data
        ]

        # Insert in batches
        for i in tqdm(range(0, len(records), 1000), desc="Inserting ICD-10 codes"):
            batch = records[i:i + 1000]
            await conn.executemany(insert_query, batch)

        # Rebuild the vector index now that data exists (an ivfflat index
        # built on an empty table has degenerate clusters)
        print("\nRebuilding vector index...")
        await rebuild_icd10_vector_index(conn)

        # Verify
        count = await conn.fetchval("SELECT COUNT(*) FROM icd10_codes")
        print(f"\nSuccessfully loaded {count} ICD-10 codes")

        # Sample query
        sample = await conn.fetchrow("""
            SELECT icd10_code, description, chapter, block
            FROM icd10_codes
            WHERE chapter IS NOT NULL
            LIMIT 1
        """)
        print(f"\nSample record:")
        print(f"   Code: {sample['icd10_code']}")
        print(f"   Chapter: {sample['chapter']}")
        print(f"   Block: {sample['block']}")
        print(f"   Description: {sample['description'][:80]}...")

    finally:
        await conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only load the first N codes (for a quick validation run)")
    args = parser.parse_args()
    asyncio.run(load_icd10_codes(limit=args.limit))
