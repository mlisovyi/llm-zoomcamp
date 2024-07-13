# %%
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import requests

model_name = "multi-qa-distilbert-cos-v1"
embedding_model = SentenceTransformer(model_name)

# %%
user_question = "I just discovered the course. Can I still join it?"
v = embedding_model.encode(user_question)


# %%
base_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"
relative_url = "03-vector-search/eval/documents-with-ids.json"
docs_url = f"{base_url}/{relative_url}?raw=1"
docs_response = requests.get(docs_url)
documents = docs_response.json()

relative_url = "03-vector-search/eval/ground-truth-data.csv"
ground_truth_url = f"{base_url}/{relative_url}?raw=1"
df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == "machine-learning-zoomcamp"]
ground_truth = df_ground_truth.to_dict(orient="records")

# %%
documents = [d for d in documents if d["course"] == "machine-learning-zoomcamp"]
print(len(documents))
# %%
embeddings = []
for d in tqdm(documents, smoothing=0):
    qa_text = f"{d['question']} {d['text']}"
    embeddings.append(embedding_model.encode(qa_text))
X = np.array(embeddings)


# %%
class VectorSearchEngine:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [self.documents[i] for i in idx]


def hit_rate(relevance_total: list[list[bool]]) -> float:
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)


# %%


# %%
relevance_total = []

search_engine = VectorSearchEngine(documents=documents, embeddings=X)
for q in tqdm(ground_truth, smoothing=0):
    doc_id = q["document"]
    v_question = embedding_model.encode(q["question"])
    results = search_engine.search(v_question, num_results=5)
    relevance = [d["id"] == doc_id for d in results]
    relevance_total.append(relevance)

# %%
print(hit_rate(relevance_total))

# %%        ES
from elasticsearch import Elasticsearch

es_client = Elasticsearch("http://localhost:9200")

es_client.info()

# %%
index_settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "text_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}
index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

# %%
for i, doc in tqdm(enumerate(documents), smoothing=0):
    doc["text_vector"] = X[i]
    try:
        es_client.index(index=index_name, document=doc)
    except Exception as e:
        print(e)


# %%
def elastic_search_knn(field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        # "filter": {"term": {"course": course}},
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"],
    }

    es_results = es_client.search(index=index_name, body=search_query)

    result_docs = []

    for hit in es_results["hits"]["hits"]:
        result_docs.append(hit["_source"])

    return result_docs


v_q = embedding_model.encode(user_question)
elastic_search_knn("text_vector", v_q, "")


# %%
def es_vector_knn(q):
    question = q["question"]

    v_q = embedding_model.encode(question)

    return elastic_search_knn("text_vector", v_q, "")


relevance_total_ = []

for q in tqdm(ground_truth, smoothing=0):
    doc_id = q["document"]
    results = es_vector_knn(q)
    relevance = [d["id"] == doc_id for d in results]
    relevance_total_.append(relevance)

# %%
hit_rate(relevance_total)

# %%


# %%
