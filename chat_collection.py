#Text-embeddings modeli için: http://docker host IP:8082  (BGE-large 1024-d)
#qdrant:  http://docker host IP:6333 
#metinden metin üreten model için: ttp://docker host IP::8081/v1
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.langchain import LangchainEmbedding

from langchain_core.embeddings import Embeddings
import requests, numpy as np
from typing import List, Union
from openai import OpenAI

# ------------ TEI Embeddings (LangChain arayüzü + L2-norm) ------------
TEI_BASE_URL = "http://10.1.100.45:8082"

class TEIEmbeddings(Embeddings):
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
        if isinstance(js, list):
            return js  # [[...], [...]]
        if isinstance(js, dict) and "data" in js:
            return [row["embedding"] for row in js["data"]]
        if isinstance(js, dict) and "embeddings" in js:
            return js["embeddings"]
        raise ValueError(f"Unknown TEI response schema: {type(js)}")

    def _l2_normalize(self, vecs: List[List[float]]) -> List[List[float]]:
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).tolist()

    # ---- LangChain Embeddings API ----
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            js = self._post(self.embed_url, {"inputs": texts, "truncate": True})
        except Exception:
            js = self._post(self.embed_v1, {"input": texts})
        return self._l2_normalize(self._normalize_schema(js))

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# ------------ Chat (LlamaFile/OpenAI uyumlu endpoint) ------------
chat_client = OpenAI(base_url="http://10.1.100.45:8081/v1", api_key="x")

# === Ayarlar: TEI’yi LlamaIndex’e tak ===
Settings.embed_model = LangchainEmbedding(TEIEmbeddings(TEI_BASE_URL))
print("embed_model = TEI ok")

# === Qdrant bağlantısı ===
qdrant_host = "10.1.100.45"
qdrant_port = 6333
collection_name = "nestle"
client = QdrantClient(host=qdrant_host, port=qdrant_port)

# === Koleksiyon kontrolü ===
if not client.collection_exists(collection_name):
    print(f"Collection not found: {collection_name}")
    raise SystemExit(1)

print(f"Collection found: {collection_name}, bağlanılıyor...")

# === Vector Store & Index ===
vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Var olan koleksiyondan index yükle
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# === Soru-Cevap Döngüsü ===
while True:
    query = input("Soru: ").strip()
    if not query:
        print("Lütfen bir soru yazın.")
        continue
    if query.lower() in ("exit", "quit", "bye"):
        break

    try:
        retriever = index.as_retriever()
        relevant_nodes = retriever.retrieve(query)
        context = "\n\n".join([n.get_content() for n in relevant_nodes]) or "(no context)"

        prompt = f"""Answer the question based on the context below.
If the question is unrelated to the context, say "It is not related".

Context:
{context}

Question: {query}

Answer:"""

        response = chat_client.chat.completions.create(
            model="local",
            temperature=0.1,
            max_tokens=600,
            stop=["</s>"],
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content.strip()
        print(answer)

    except Exception as e:
        print("Hata:", e)
        break
