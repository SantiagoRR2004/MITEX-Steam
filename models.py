from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(".env.local")


modelName = "google/gemma-3-1b-it"

EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")
GEN_TOKENIZER = AutoTokenizer.from_pretrained(modelName)
GEN_MODEL = AutoModelForCausalLM.from_pretrained(modelName)
