import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from ingest import get_embedding_function, VECTOR_DB_DIR

load_dotenv()

PROMPT_TEMPLATE = """
You are an enterprise document intelligence assistant. Answer the user's question strictly using ONLY the provided context below. 

Rules:
1. If the exact answer cannot be derived directly from the context, state clearly: "I cannot find this information in the provided document."
2. Do NOT use outside knowledge or extrapolate beyond the text.
3. Keep your response professional, precise, and structured.

Context:
{context}

Question: {question}

Answer:
"""

class RAGEngine:
    def __init__(self):
        self.embeddings = get_embedding_function()
        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embeddings
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash-lite",
            temperature=0.0,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

    # Increased default context window to 10 chunks
    def query(self, question: str, k: int = 10) -> Dict[str, Any]:
        results = self.vector_db.similarity_search_with_score(question, k=k)

        if not results:
            return {
                "answer": "No relevant documents found in index.",
                "citations": []
            }

        context_blocks = []
        citations = []

        for doc, score in results:
            context_blocks.append(doc.page_content)
            citations.append({
                "source": os.path.basename(doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page", 0) + 1,
                "relevance_score": round(float(score), 4),
                "excerpt": doc.page_content[:150] + "..."
            })

        formatted_context = "\n\n---\n\n".join(context_blocks)
        
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        messages = prompt.format_messages(context=formatted_context, question=question)
        
        response = self.llm.invoke(messages)

        # Extract clean text string from response.content
        answer_text = response.content
        if isinstance(answer_text, list):
            answer_text = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in answer_text
            )
        elif not isinstance(answer_text, str):
            answer_text = str(answer_text)

        return {
            "answer": answer_text,
            "citations": citations
        }