# pip install llama-index llama-index-vector-stores-qdrant llama-index-embeddings-langchain
# pip install qdrant-client
# pip install requests numpy
# (TEI için: ghcr.io/huggingface/text-embeddings-inference imajı ve modeliniz 8082'de çalışıyor varsayılmıştır)

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, Settings, VectorStoreIndex
from llama_index.readers.file.docs.base import PDFReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding

from langchain_core.embeddings import Embeddings
import requests
import numpy as np
from typing import List, Union
import os

# ------------------- TEI Embeddings (LangChain Embeddings + L2-norm) -------------------
TEI_BASE_URL = "http://10.1.100.45:8082"  # TEI endpoint'in

class TEIEmbeddings(Embeddings):
    """LangChain Embeddings arayüzü + L2-normalize (cosine için).
       /embed bazı sürümlerde liste [[...]] döndürebildiğinden şema esnekliği sağlar."""
    def __init__(self, base_url: str, timeout: int = 60):
        self.embed_url = base_url.rstrip("/") + "/embed"          # TEI native
        self.embed_v1  = base_url.rstrip("/") + "/v1/embeddings"  # OpenAI uyumlu
        self.sess = requests.Session()
        self.timeout = timeout

    def _post(self, url: str, payload: dict):
        r = self.sess.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _normalize_schema(self, js: Union[list, dict]) -> List[List[float]]:
        # 1) /embed → [[...], [...]]
        if isinstance(js, list):
            return js
        # 2) {"data":[{"embedding":[...]}]}
        if isinstance(js, dict) and "data" in js:
            return [row["embedding"] for row in js["data"]]
        # 3) {"embeddings":[[...],[...]]}
        if isinstance(js, dict) and "embeddings" in js:
            return js["embeddings"]
        raise ValueError(
            f"Unknown TEI response schema type={type(js)} "
            f"keys={list(js.keys()) if isinstance(js, dict) else 'n/a'}"
        )

    def _l2_normalize(self, vecs: List[List[float]]) -> List[List[float]]:
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).tolist()

    # ---- LangChain Embeddings API ----
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Önce /embed, olmazsa /v1/embeddings
        try:
            js = self._post(self.embed_url, {"inputs": texts, "truncate": True})
        except Exception:
            js = self._post(self.embed_v1, {"input": texts})
        vecs = self._normalize_schema(js)
        return self._l2_normalize(vecs)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# ------------------- Qdrant bağlantısı -------------------
client = QdrantClient(host="10.1.100.45", port=6333)

# TEI (BGE-large) ~1024-d → Qdrant koleksiyonu buna göre oluşturulur
EMBED_DIM = 1024

# LlamaIndex global embedding modeli: TEI + L2-norm
Settings.embed_model = LangchainEmbedding(TEIEmbeddings(TEI_BASE_URL))
print("embed_model = TEI ok")

# ------------------- Koleksiyon yardımcıları -------------------
def ensure_collection(client: QdrantClient, collection_name: str):
    if client.collection_exists(collection_name):
        print("Koleksiyon zaten var")
        return False, f"Koleksiyon zaten var: {collection_name}"
    else:
        print("Koleksiyon yok → oluşturuluyor")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print("Koleksiyon üretildi")
        return True, f"Yeni koleksiyon oluşturuldu: {collection_name}"

# ------------------- PDF yükleme akışı -------------------
def upload_pdf_to_qdrant(file: str, collection_name: str):
    try:
        if not os.path.isfile(file) or not file.endswith(".pdf"):
            return "PDF dosyası gerekli."

        created, msg = ensure_collection(client, collection_name)
        if created is False:
            print(msg)

        # 1) PDF oku
        documents = PDFReader().load_data(file)
        print("documents ok")

        # 2) Chunk'lara böl
        node_parser = SimpleNodeParser(chunk_size=512, chunk_overlap=50)
        print("parser ok")

        # 3) Vector store + storage context
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        print("vector_store & storage_context ok")

        # 4) Indexle (embedding TEI'den gelecek)
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            node_parser=node_parser,
            show_progress=True,
        )
        print("VectorStoreIndex ok")

        return f"'{file}' dosyası '{collection_name}' koleksiyonuna yüklendi."

    except Exception as e:
        return f"Hata: {e}"

# ------------------- CLI -------------------
if __name__ == "__main__":
    file = input("file: ").strip()
    if file not in os.listdir(os.getcwd()) or not file.endswith(".pdf"):
        print("PDF dosyası gerekli.")
    else:
        collection = file[:-4]
        out = upload_pdf_to_qdrant(file, collection)
        print(out)
