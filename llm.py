from transformers import pipeline
from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()

MODEL_NAME = "google/flan-t5-base"

# ─────────────────────────────
# Hugging Face backend
# ─────────────────────────────
def load_huggingface_llm():
    """
    Load and return the Hugging Face LLM pipeline.
    """

    return pipeline(
        "text2text-generation",
        model=MODEL_NAME,
        max_new_tokens=256,
        temperature=0.0,
        
    )

# ─────────────────────────────
# OpenAI backend
# ─────────────────────────────
def load_openai_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    def openai_llm(prompt: str):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a careful assistant that follows instructions exactly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        return [{
            "generated_text": response.choices[0].message.content
        }]

    return openai_llm

# ─────────────────────────────
# Dispatcher (unchanged interface)
# ─────────────────────────────
def load_llm():
    """
    Load and return the configured LLM backend.

    Defaults to Hugging Face unless LLM_BACKEND=openai is set.
    """
    backend = os.getenv("LLM_BACKEND", "huggingface").lower()

    if backend == "openai":
        return load_openai_llm()

    return load_huggingface_llm()

def load_router_llm():
    """
    Load a lightweight, local LLM for routing decisions.
    Always uses Hugging Face for deterministic behavior.
    """
    return load_huggingface_llm()


def generate_answer(question: str, context: str, llm) -> str:
    """
    Generate an answer to the given question using only the provided context.

    If the answer cannot be found in the context, the model is instructed
    to respond that it does not know.
    """
    prompt = f"""
You are a helpful assistant answering questions using the provided context.

Answer the question directly.
Use the context only to support your answer.
Ignore information that is not relevant to the question.
If the answer is not contained in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
    response = llm(prompt)
    return response[0]["generated_text"].strip()


def needs_retrieval(question: str) -> bool:
    """
    Decide whether the question requires document retrieval.

    Returns True if external context is needed, False otherwise.
    """

    router_llm = load_router_llm()

    prompt = f"""
You are deciding whether a question requires looking up information
from external documents.

Answer YES if the question:
- asks how a specific software library, framework, or tool works
- asks about implementation details, configuration, or internal behavior
- mentions a specific library name (e.g., LangChain, LangGraph)

Answer NO only if the question is general knowledge, opinion-based,
or conversational and does not depend on technical documentation.

Question:
{question}

Answer with only YES or NO.
"""
    response = router_llm(prompt)[0]["generated_text"].strip().upper()
    return response.startswith("Y")
