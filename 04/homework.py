# %%
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np

base_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"
relative_url = "04-monitoring/data/results-gpt4o-mini.csv"
docs_url = f"{base_url}/{relative_url}?raw=1"

# %%
df = pd.read_csv(docs_url, nrows=300)
# %%
model_name = "multi-qa-mpnet-base-dot-v1"
embedding_model = SentenceTransformer(model_name)

# %%
answer_llm = df.iloc[0].answer_llm
v_answer_llm = embedding_model.encode(answer_llm)
print(v_answer_llm[0])

# %%
v_llm = []
v_orig = []
for i, s in tqdm(df.iterrows(), total=len(df)):
    llm = s["answer_llm"]
    orig = s["answer_orig"]
    v_llm.append(embedding_model.encode(llm))
    v_orig.append(embedding_model.encode(orig))
# %%
results = []
for v1, v2 in zip(v_llm, v_orig):
    results.append(v1.dot(v2) / np.sqrt(v1.dot(v1)) / np.sqrt(v2.dot(v2)))
pd.Series(results).describe()
# %%
from rouge import Rouge

rouge_scorer = Rouge()

scores = rouge_scorer.get_scores(df["answer_llm"], df["answer_orig"])

# %%
q4 = scores[10]["rouge-1"]["f"]
print(f"Q4: {q4:.3f}")

# %%
q5 = np.mean([v["f"] for v in scores[10].values()])
print(f"Q5: {q5:.3f}")

# %%
q6 = np.mean([v["rouge-2"]["f"] for v in scores])
print(f"Q6 = {q6:.3f}")

# %%
