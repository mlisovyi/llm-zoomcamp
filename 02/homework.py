# %%
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)
# %%
response = client.chat.completions.create(
    model="gemma:2b",
    messages=[{"role": "user", "content": "What's the formula for energy?"}],
    temperature=0.0,
)
# %%
answer = response.choices[0].message.content

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
# %%
len(tokenizer.encode(answer))

# %%
