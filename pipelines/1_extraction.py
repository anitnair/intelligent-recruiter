import spacy
import pandas as pd
from typing import List, Dict
from config.settings import MASKED_ENTITIES

# 1. Load spaCy model and ESCO skills
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Need a larger model or download the default
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load ESCO skills for matching/normalization
# Note: Ensure data/esco_skills.csv exists and has a 'skill_name' column
try:
    ESCO_SKILLS = set(pd.read_csv('data/esco_skills.csv')['skill_name'].str.lower())
except FileNotFoundError:
    print("WARNING: esco_skills.csv not found. Using mock skill list.")
    ESCO_SKILLS = {'python', 'django', 'flask', 'javascript', 'developer', 'engineer'}


def apply_guardrails(doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
    """Masks protected attributes in the spaCy document."""
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            if ent.label_ in MASKED_ENTITIES:
                # Replace the entity span with a generic, masked placeholder
                retokenizer.merge(ent, attrs={"LEMMA": "[MASKED_ENTITY]", "TAG": "XX"})
    return doc

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Runs the full extraction pipeline with normalization and guardrails."""
    doc = nlp(text)
    doc = apply_guardrails(doc) # Apply bias guardrail FIRST

    # Rule-Based Matching for Skills (using ESCO lexicon)
    matcher = spacy.matcher.PhraseMatcher(nlp.vocab)
    skill_patterns = [nlp.make_doc(skill) for skill in ESCO_SKILLS]
    if skill_patterns:
        matcher.add("SKILL", skill_patterns)

    matches = matcher(doc)

    extracted = {
        'text_clean': doc.text, # Cleaned text after masking
        'skills': [],
        'roles': [],
        'experience_text': [] # For embedding
    }

    # Gather matched skills
    for match_id, start, end in matches:
        span = doc[start:end]
        extracted['skills'].append(span.text.lower())

    # Example: Simple regex/keyword search for roles
    for token in doc:
        if token.text.lower() in ['developer', 'engineer', 'analyst']:
            extracted['roles'].append(token.text)

    # Simple aggregation of relevant sections for embedding
    # (In a real scenario, this would involve document layout parsing)
    extracted['experience_text'] = [doc.text] # Placeholder for full context

    # Deduplicate skills and convert to normalized set
    extracted['skills'] = list(set(extracted['skills']))

    return extracted

def process_all_resumes(resume_paths: List[str]) -> List[Dict]:
    """Processes all resumes and returns a list of structured records."""
    structured_data = []

    # Mock data processing
    for i in range(len(resume_paths)):
        # Example text containing maskable entities
        text = "Experienced Python Developer with 5 years experience in Django and Flask. My name is Alex Johnson, and I graduated in 2015."
        data = extract_entities(text)
        data['candidate_id'] = f"CAND_{i:04d}"
        structured_data.append(data)

    return structured_data

if __name__ == '__main__':
    # Example usage
    mock_paths = ["resume1.txt", "resume2.txt"]
    data = process_all_resumes(mock_paths)
    print(f"Processed {len(data)} records.")
    # Show how the guardrail worked (name Alex Johnson and date 2015 should be masked)
    print(data[0])