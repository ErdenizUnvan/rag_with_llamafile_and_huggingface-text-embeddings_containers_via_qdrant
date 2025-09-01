#pip install qdrant_client
from qdrant_client import QdrantClient

# Qdrant'a bağlan
client = QdrantClient(url="http://10.1.100.45:6333")

# Koleksiyonların listesini al
collections = client.get_collections()
if len(collections.collections)==0:
    print('collection yok zaten')
else:
    for col in collections.collections:
        print(f"{col.name}")
