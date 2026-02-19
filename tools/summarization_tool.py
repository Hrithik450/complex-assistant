import tiktoken
from langchain.tools import tool
from typing import List

# --- NEW: Import LLM for advanced analysis ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# --- LLM and Prompt Initialization ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=os.getenv("GOOGLE_API_KEY"))
encoding = tiktoken.get_encoding("cl100k_base")

# Prompt for the "Map" step (summarizing a single chunk)
map_prompt_template = ChatPromptTemplate.from_template(
    """
    You are an expert summarizer. The following text is a small chunk of a larger email conversation.
    Your task is to concisely summarize the key points, decisions, and action items from this specific chunk.
    Focus only on the information present in this text.

    Text chunk:
    ---
    {chunk}
    ---

    Concise summary of the chunk:
    """
)
map_chain = map_prompt_template | llm | StrOutputParser()

# Prompt for the "Reduce" step (combining summaries)
reduce_prompt_template = ChatPromptTemplate.from_template(
    """
    You are an expert at synthesizing information. The following are multiple summaries from different parts of a single, long email conversation.
    Your task is to combine them into one final, cohesive, and comprehensive summary of the entire conversation.
    Ensure the final summary flows logically and captures the overall narrative, key outcomes, and any unresolved issues.

    Individual summaries:
    ---
    {chunk_summaries}
    ---

    Final comprehensive summary:
    """
)
reduce_chain = reduce_prompt_template | llm | StrOutputParser()


def get_text_chunks(text: str, chunk_size_tokens: int = 3000) -> List[str]:
    """Splits text into chunks based on token count."""
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size_tokens):
        chunk_tokens = tokens[i:i + chunk_size_tokens]
        chunks.append(encoding.decode(chunk_tokens))
    return chunks


@tool("summarization_tool", parse_docstring=True)
def summarization_tool(email_content: str) -> str:
    """
    Summarizes very long email conversations or documents that may exceed the context window of a language model.
    It uses a Map-Reduce technique to handle large amounts of text efficiently.

    Args:
        email_content (str): The full text content of the email thread or document to be summarized.
    """
    if not email_content.strip():
        return "Error: No content provided to summarize."

    print("Summarization tool called. Applying Map-Reduce strategy.")

    # 1. Split the content into manageable chunks
    chunks = get_text_chunks(email_content)

    # 2. MAP step: Summarize each chunk individually
    # In a production system, you might run these in parallel (e.g., with asyncio.gather)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summary = map_chain.invoke({"chunk": chunk})
        chunk_summaries.append(summary)

    # 3. REDUCE step: Combine the summaries into a final summary
    print("Combining chunk summaries into a final summary...")
    combined_summaries = "\n\n---\n\n".join(chunk_summaries)
    final_summary = reduce_chain.invoke({"chunk_summaries": combined_summaries})

    return final_summary