# %%
import json

import tiktoken
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

# %%
with open("documents.json", "rt") as f:
    docs_raw = json.load(f)

docs = []

for course in docs_raw:
    for doc in course["documents"]:
        doc["course"] = course["course"]
        docs.append(doc)

# %%

esc = Elasticsearch("http://localhost:9200")
index_settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
        }
    },
}
esc.indices.create(index="dummy", body=index_settings)

for doc in tqdm(docs):
    esc.index(index="dummy", document=doc)

# %%
q = "How do I execute a command in a running docker container?"
search_query = {
    "size": 3,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": q,
                    "fields": ["question^4", "text"],
                    "type": "best_fields",
                }
            },
            "filter": {"term": {"course": "machine-learning-zoomcamp"}},
        }
    },
}
res = esc.search(index="dummy", body=search_query)
print(res["hits"]["hits"][-1])
# %%
context_template = """
Q: {question}
A: {text}
""".strip()
context = ""
for d in res["hits"]["hits"]:
    doc = d["_source"]
    context += (
        context_template.format(question=doc["question"], text=doc["text"]) + "\n\n"
    )
context = context.strip()
print(context)
# %%
question = "How do I execute a command in a running docker container?"
prompt = f"""
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

print(len(prompt))
# %%
encoding = tiktoken.encoding_for_model("gpt-4o")
tokens = encoding.encode(prompt)
len(tokens)
# %%
