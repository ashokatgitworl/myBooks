import chromadb
from paths import VECTOR_DB_DIR
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
# --- List all collections ---
collections = client.list_collections()
print("📚 Available Collections:")
for col in collections:
    print(f" - {col.name}")

# --- Read specific collection ---
collection_name = "publications"  # change this to your collection name
try:
    collection = client.get_collection(name=collection_name)
    print(f"\n✅ Connected to collection: {collection_name}")

    # --- View some data ---
    results = collection.get()  # get all documents and embeddings
    print(f"\n🧾 Collection contents: {results}")
except Exception as e:
    print(f"⚠️ Error: {e}")
