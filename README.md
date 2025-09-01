# rag_with_llamafile_and_huggingface-text-embeddings_containers_via_qdrant

@for docker host:

docker pull eunvan/bge-tei:1.8

docker run -d --rm --name tei-bge
-p 8082:80
eunvan/bge-tei:1.8

docker pull eunvan/qwen2.5-llamafile:latest

docker container run --name my-rag-llm -d --rm -p 8081:8080 eunvan/qwen2.5-llamafile:latest

docker volume create qdrant_storage

docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant

@for windows pc:

pip install -r requirements.txt

python add_file.py

nestle.pdf

python chat_collection.py



