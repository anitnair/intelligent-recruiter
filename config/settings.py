# Configuration for the Recruitment RAG MVP

# --- Neo4j Connection ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"
KG_DATABASE = "recruitment_db"

# --- Model & Data Constants ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 512 # Max token length for embedding/chunking
OVERLAP = 50

# --- Guardrails Configuration ---
# List of entities to mask/remove during parsing (Step 1)
MASKED_ENTITIES = [
    'PERSON', 'GENDER', 'AGE', 'DATE', 'ADDRESS', 'ETHNICITY'
]

# --- LLM Settings (Assuming Llama 3 8B via API/Inference Server) ---
LLM_API_ENDPOINT = "http://localhost:8000/v1/generate"
LLM_SYSTEM_PROMPT = (
    "You are an expert Talent Analyst. Generate a professional match rationale. "
    "Your response MUST only be based on the provided skills and experience. "
    "DO NOT mention names, age, or any protected characteristics."
)