from pathlib import Path
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_QDRANT_PATH = PROJECT_ROOT / "data" / "vector_store" / "qdrant"

COLLECTION_NAME = "omnimind_documents"
VECTOR_SIZE = 384

SERVER_URL = "http://127.0.0.1:6333"


def main():
    print("=" * 70)
    print("OmniMind Qdrant Migration")
    print("=" * 70)

    print(f"Local source : {LOCAL_QDRANT_PATH}")
    print(f"Server target: {SERVER_URL}")
    print(f"Collection   : {COLLECTION_NAME}")
    print()

    # ---------------------------------------------------------
    # Connect to the existing LOCAL Qdrant database
    # ---------------------------------------------------------
    print("1. Opening existing local Qdrant...")

    local = QdrantClient(
        path=str(LOCAL_QDRANT_PATH)
    )

    collections = local.get_collections().collections

    if not any(c.name == COLLECTION_NAME for c in collections):
        local.close()
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' was not found in local Qdrant."
        )

    local_info = local.get_collection(COLLECTION_NAME)

    local_count = local.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print(f"   Local vectors: {local_count}")

    if local_count == 0:
        local.close()
        raise RuntimeError("Local Qdrant contains zero vectors.")

    # ---------------------------------------------------------
    # Connect to SERVER Qdrant
    # ---------------------------------------------------------
    print()
    print("2. Connecting to Qdrant server...")

    server = QdrantClient(
        url=SERVER_URL
    )

    print("   Server connection: OK")

    # ---------------------------------------------------------
    # Create collection if necessary
    # ---------------------------------------------------------
    print()
    print("3. Preparing server collection...")

    server_collections = server.get_collections().collections

    if not any(c.name == COLLECTION_NAME for c in server_collections):
        print("   Creating collection...")

        server.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

    else:
        print("   Collection already exists.")

    # ---------------------------------------------------------
    # Read ALL points from local Qdrant
    # ---------------------------------------------------------
    print()
    print("4. Reading local vectors...")

    points = []

    offset = None

    while True:
        batch, offset = local.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        points.extend(batch)

        if offset is None:
            break

    print(f"   Points loaded: {len(points)}")

    if len(points) != local_count:
        local.close()
        server.close()
        raise RuntimeError(
            f"Point count mismatch while reading local Qdrant: "
            f"expected {local_count}, got {len(points)}"
        )

    # ---------------------------------------------------------
    # Upload points to server
    # ---------------------------------------------------------
    print()
    print("5. Uploading vectors to Qdrant server...")

    from qdrant_client.models import PointStruct

    upload_points = []

    for point in points:
        upload_points.append(
            PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )
        )

    # Upload in batches
    batch_size = 100

    for start in range(0, len(upload_points), batch_size):
        batch = upload_points[start:start + batch_size]

        server.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True,
        )

        print(
            f"   Uploaded {min(start + batch_size, len(upload_points))}"
            f"/{len(upload_points)}"
        )

    # ---------------------------------------------------------
    # Verify
    # ---------------------------------------------------------
    print()
    print("6. Verifying migration...")

    server_count = server.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print(f"   Local vectors : {local_count}")
    print(f"   Server vectors: {server_count}")

    if server_count != local_count:
        local.close()
        server.close()
        raise RuntimeError(
            f"Migration verification failed: "
            f"local={local_count}, server={server_count}"
        )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    local.close()
    server.close()

    print()
    print("=" * 70)
    print("MIGRATION SUCCESSFUL")
    print("=" * 70)
    print(f"Vectors migrated: {server_count}")
    print("Original local Qdrant database was NOT modified.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("MIGRATION FAILED")
        print(f"Error: {exc}")
        sys.exit(1)