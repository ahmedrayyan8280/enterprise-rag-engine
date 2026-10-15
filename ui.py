import streamlit as st
import tempfile
import os
from ingest import process_and_index_pdf
from rag_engine import RAGEngine
from st_copy_to_clipboard import st_copy_to_clipboard

st.set_page_config(
    page_title="DocMind Enterprise RAG",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 DocMind: Enterprise Document RAG Engine")
st.caption("Context-grounded document intelligence with strict anti-hallucination guardrails and page citations.")

# Sidebar Configuration
st.sidebar.header("📁 Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload an Enterprise PDF...", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Index Document", type="primary"):
        with st.spinner("Chunking text & generating vector embeddings via Google Cloud..."):
            
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                chunks = process_and_index_pdf(file_path)
            
            st.sidebar.success(f"Indexed {chunks} semantic chunks into ChromaDB!")

st.sidebar.markdown("---")
st.sidebar.markdown("**System Architecture:**")
st.sidebar.markdown("- **Embeddings:** Google Cloud (gemini-embedding-2)")
st.sidebar.markdown("- **Vector Store:** ChromaDB (HNSW)")
st.sidebar.markdown("- **LLM:** Google Gemini 3.8 Flash")

# Main Query Interface
query = st.text_input("Ask a question about your indexed document:", placeholder="e.g., What are the key findings?")

if query:
    with st.spinner("Retrieving vector contexts & generating response..."):
        try:
            engine = RAGEngine()
            results = engine.query(query, k=10)

            st.subheader("💡 Answer")
            st.write(results["answer"])

            st_copy_to_clipboard(
                text=results["answer"],
                before_copy_label="📋 Copy Answer",
                after_copy_label="✅ Copied to Clipboard!"
            )

            st.markdown("---")
            st.subheader("📌 Retrieved Source Citations")

            if results["citations"]:
                for i, citation in enumerate(results["citations"], 1):
                    with st.expander(f"Source {i}: {citation['source']} (Page {citation['page']}) — Distance: {citation['relevance_score']}"):
                        st.write(f"**Text Excerpt:**")
                        st.info(f'"{citation["excerpt"]}"')
            else:
                st.warning("No context chunks retrieved.")

        except Exception as e:
            # Gracefully catch Google API overloads
            error_msg = str(e)
            if "503" in error_msg or "UNAVAILABLE" in error_msg or "high demand" in error_msg:
                st.warning("⚠️ **Google AI servers are currently experiencing high demand.** Please wait a few seconds and try again.")
            else:
                st.error(f"⚠️ **An unexpected error occurred:** {error_msg}")