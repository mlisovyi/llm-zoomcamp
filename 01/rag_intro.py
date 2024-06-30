# %%
import minsearch
import json
from openai import OpenAI

# %%
with open("documents.json", "rt") as f:
    docs_raw = json.load(f)

# %%
docs = []

for course in docs_raw:
    for doc in course["documents"]:
        doc["course"] = course["course"]
        docs.append(doc)
# %%
index = minsearch.Index(
    text_fields=["text", "section", "question"], keyword_fields=["course"]
)

# %%
index.fit(docs)


# %%
def search(query, index: minsearch.Index):
    boosting = {"question": 3, "section": 0.5}
    search_results = index.search(
        query=query,
        boost_dict=boosting,
        num_results=5,
        filter_dict={"course": "data-engineering-zoomcamp"},
    )

    return search_results


def build_prompt(query, search_results):
    prompt_template = """
You are a course aching assistant. Answer the QUESTION based on the CONTEXT.
Use only facts from CONTEXT, when answering the QUESTION.
If the CONTEXT doesn't contain the answer, output NONE.
QUESTION: {query}
CONTEXT:
{context}
""".strip()

    context = ""
    for doc in search_results:
        context += f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    prompt = prompt_template.format(query=query, context=context).strip()
    return prompt


def llm(prompt):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    return answer


# %%
query = "how do I run kafka?"
search_results = search(query, index)
prompt = build_prompt(query, search_results)
print(prompt)

# %%
answer = llm(prompt)

# %%
answer
