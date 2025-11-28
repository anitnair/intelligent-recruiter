from neo4j import GraphDatabase
from typing import Dict, List
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, KG_DATABASE

class KGBuilder:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            print("Neo4j connection established.")
        except Exception as e:
            print(f"ERROR: Could not connect to Neo4j. Check settings.py and Neo4j status. {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def load_candidate_data(self, structured_data: List[Dict]):
        """Loads structured candidate data into Neo4j."""
        if not self.driver: return

        with self.driver.session(database=KG_DATABASE) as session:
            for record in structured_data:
                candidate_id = record['candidate_id']

                # 1. Create Candidate Node
                # Only storing clean_text and ID (no sensitive attributes) - Guardrail
                session.run(
                    "MERGE (c:Candidate {id: $id, clean_text: $text})",
                    id=candidate_id, text=record['text_clean']
                )

                # 2. Create Skill Nodes and HAS_SKILL relationship
                for skill in record['skills']:
                    session.run(
                        """
                        MERGE (s:Skill {name: $skill_name})
                        MERGE (c:Candidate {id: $id})
                        MERGE (c)-[:HAS_SKILL]->(s)
                        """,
                        skill_name=skill, id=candidate_id
                    )

                print(f"Loaded Candidate {candidate_id} and {len(record['skills'])} skills.")

    # Placeholder for Job loading for Hybrid Search
    def load_job_data(self):
        """Mock loading a Job Node and its required skills."""
        if not self.driver: return
        with self.driver.session(database=KG_DATABASE) as session:
            session.run("MERGE (j:Job {id: 'target_job', title: 'Senior Python Developer'})")
            session.run(
                """
                MATCH (j:Job {id: 'target_job'})
                MERGE (s1:Skill {name: 'python'})
                MERGE (s2:Skill {name: 'django'})
                MERGE (j)-[:REQUIRES_SKILL]->(s1)
                MERGE (j)-[:REQUIRES_SKILL]->(s2)
                """
            )
        print("Mock Job loaded.")


if __name__ == '__main__':
    from pipelines.1_extraction import process_all_resumes

    # 1. Get structured data from Step 2
    mock_data = process_all_resumes(["mock.txt"] * 5)

    # 2. Load into KG
    builder = KGBuilder()
    if builder.driver:
        builder.load_candidate_data(mock_data)
        builder.load_job_data()
    builder.close()
    print("Graph loading complete.")# kg builder