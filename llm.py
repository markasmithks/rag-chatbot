from transformers import pipeline

MODEL_NAME = "google/flan-t5-base"


def load_llm():
    """
    Load and return the LLM pipeline.

    This function is intentionally minimal so it can be cached
    by the caller (e.g., Streamlit) and swapped out later if needed.
    """
    return pipeline(
        "text2text-generation",
        model=MODEL_NAME,
        max_new_tokens=256,
    )


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
