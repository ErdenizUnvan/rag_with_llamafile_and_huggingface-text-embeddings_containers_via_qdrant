#qdrant:  http://docker host IP:6333 
from qdrant_client import QdrantClient

# Qdrant'a bağlan
client = QdrantClient(host="10.1.100.45", port=6333)

# Silmek istediğin koleksiyon
collection_name = input('collection_name: ')

# Varsa sil
if client.collection_exists(collection_name):
    client.delete_collection(collection_name=collection_name)
    print(f"'{collection_name}' koleksiyonu silindi.")
else:
    print(f"'{collection_name}' koleksiyonu zaten yok.")
