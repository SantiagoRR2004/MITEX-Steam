"""
First:
curl -fsSL https://ollama.com/install.sh | sh
Then:
ollama run gemma4:e2b
"""

from ollama import chat

response = chat(
    model="gemma4:e2b",
    messages=[{"role": "user", "content": "What is a RAG?"}],
)
print(response.message.content)
