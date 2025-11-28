import streamlit as st
import sys
import os
from typing import Dict, List

# Add the project root and pipelines directory to the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pipelines')))

# Conditional import after path manipulation
try:
    from pipelines.3_rag_service import RAGService
except ImportError as e:
    st.error(f"Failed to import RAGService. Check your file paths and dependencies. Error: {e}")
    # Define a mock class if import fails to prevent immediate crash
    class RAGService:
        def __init__(self): self.initialized = False
        def close(self): pass
        def hybrid_candidate_search(self, query: str, top_k: int = 5) -> List[Dict]: return []
        def generate_rationale(self, retrieved_candidates: List[Dict], job_query: str) -> List[Dict]: return []


# --- Initialize Service (Run once) ---
@st.cache_resource
def get_rag_service():
    """Initializes the RAGService and caches the heavy models."""
    service = RAGService()
    if not service.initialized:
        st.warning("RAG Service failed to initialize. Please ensure Neo4j is running and dependencies are met.")
    return service

rag_service = get_rag_service()

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🧩 Intelligent Recruitment Assistant (MVP)")
st.caption("Hybrid Search + RAG for Explainable and Bias-Mitigated Matchmaking")

st.markdown("---")

if rag_service.initialized:
    job_query = st.text_area(
        "Enter Job Description or Recruiter Query:",
        "We need a Senior Python Developer with deep experience in Django and Flask, 5+ years preferred."
    )

    top_k = st.slider("Number of Candidates to Retrieve", 1, 10, 5)

    if st.button("Find and Explain Candidates", type="primary"):
        with st.spinner("1. Executing Hybrid Search... (Vector & Graph Retrieval)"):
            # Step 1: Hybrid Retrieval
            retrieved_candidates = rag_service.hybrid_candidate_search(job_query, top_k=top_k)

        if retrieved_candidates:
            with st.spinner("2. Generating Rationale with LLM... (Applying Bias Guardrails)"):
                # Step 2: Generation with Guardrails
                final_results = rag_service.generate_rationale(retrieved_candidates, job_query)

            st.success(f"Found and analyzed {len(final_results)} candidates.")

            st.subheader("Ranked Candidates with Explainable Rationale")

            # Display Results
            for i, result in enumerate(final_results):
                # Use columns for clear, compact display
                col1, col2 = st.columns([1, 4])

                with col1:
                    st.metric(label=f"Rank {i+1}", value=f"Candidate: {result['candidate_id']}")

                    # Ensure score is numeric before passing to progress bar
                    score = result.get('hybrid_score', 0)
                    st.progress(float(score) if score else 0.0)
                    st.markdown(f"**Score:** `{score:.4f}`")

                with col2:
                    st.info(f"**AI Match Rationale:**")
                    st.write(result.get('rationale', 'Rationale generation failed.'))

                    # For transparency, show the context chunk the LLM used
                    with st.expander("Show Retrieved Context Chunk (The source document)"):
                        st.code(result.get('context_chunk', 'N/A'), language='text')

                st.markdown("---")
        else:
            st.warning("No candidates were returned by the hybrid search. Check data loading, graph filters, or Neo4j status.")

else:
    st.error("System is not operational. Please check `config/settings.py` and ensure external services (Neo4j, LLM) are running.")
