from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import requests
import json
from typing import Dict, List
from config.settings import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, KG_DATABASE,
    EMBEDDING_MODEL_NAME, LLM_API_ENDPOINT, LLM_SYSTEM_PROMPT
)

class RAGService:
    def __init__(self):
        try:
            # 1. Initialize Neo4j Driver
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.driver.verify_connectivity()

            # 2. Initialize Embedding Model
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Embedding model loaded.")
            self.initialized = True
        except Exception as e:
            print(f"ERROR initializing RAGService: {e}")
            self.initialized = False
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def hybrid_candidate_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Executes hybrid (Vector + Graph) search to retrieve top candidates.

        Note: This MVP query performs a *filtered* graph search for simplicity
        and mocks the vector similarity score.
        """
        if not self.initialized: return []

        query_vector = self.embedding_model.encode(query).tolist()

        # Cypher Query (Mocks Hybrid Search)
        # 1. Filters candidates who HAVE the skills required by 'target_job' (Graph Guardrail)
        # 2. Returns the candidate context (clean_text)
        cypher_query = """
        MATCH (j:Job {id: 'target_job'})-[:REQUIRES_SKILL]->(rs:Skill)
        MATCH (c:Candidate)-[:HAS_SKILL]->(rs)
        
        // This calculates a dummy score. In a real system, GDS would calculate similarity
        WITH DISTINCT c
        
        // Mocking the score for MVP: High score if they match required skills
        // and using a random component for ranking variability.
        RETURN 
            c.id AS candidate_id, 
            c.clean_text AS context_chunk, 
            (0.8 + rand() * 0.1) AS hybrid_score
        ORDER BY hybrid_score DESC
        LIMIT $k
        """

        with self.driver.session(database=KG_DATABASE) as session:
            try:
                results = session.run(cypher_query, k=top_k)
                return [dict(record) for record in results]
            except Exception as e:
                print(f"Error executing Neo4j search: {e}")
                return []

    def generate_rationale(self, retrieved_candidates: List[Dict], job_query: str) -> List[Dict]:
        """Generates a rationale for each candidate using the LLM and system guardrail."""

        for candidate in retrieved_candidates:
            # Construct the augmented prompt
            context = f"Candidate ID: {candidate['candidate_id']}\n---\nProfile Context: {candidate['context_chunk']}\n---"

            prompt_content = (
                f"Job Query: {job_query}\nCandidate Context:\n{context}\n\n"
                "Based ONLY on the context, provide a detailed rationale why this candidate is a good fit, focusing on skills and experience."
            )

            payload = {
                "systemInstruction": LLM_SYSTEM_PROMPT, # Bias Guardrail: Enforces focus on skills
                "contents": [{"parts": [{"text": prompt_content}]}]
            }

            try:
                # --- MOCK LLM CALL ---
                # In a real environment, you would use requests.post to your LLM API

                # Mock Rationale Generation (Ensures guardrail compliance by focusing on facts)
                rationale = (
                    f"Candidate {candidate['candidate_id']} is a strong match due to confirmed proficiency in both Django and Flask, "
                    "which are key Python web frameworks required by the role. The resume indicates multiple years of "
                    "developer experience, directly addressing the seniority requirement. The match is solely based on "
                    "technical qualifications and professional history."
                )

            except Exception as e:
                rationale = f"Error generating rationale (API call failed): {e}"

            candidate['rationale'] = rationale

        return retrieved_candidates

if __name__ == '__main__':
    service = RAGService()
    if service.initialized:
        job_query = "Senior Python Developer needing Django and 5+ years experience."

        # Step 1: Hybrid Retrieval
        retrieved_cands = service.hybrid_candidate_search(job_query, top_k=2)

        # Step 2: Generation with Guardrails
        final_results = service.generate_rationale(retrieved_cands, job_query)

        for result in final_results:
            print(f"\n--- Candidate: {result['candidate_id']} (Score: {result['hybrid_score']:.4f}) ---")
            print(f"Context: {result['context_chunk']}")
            print(f"Rationale: {result['rationale']}")

    service.close()