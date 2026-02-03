import streamlit as st
import ollama
from neo4j import GraphDatabase
import plotly.graph_objects as go

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Intelligent Recruitment Assistant", layout="wide")

st.markdown("""
    <style>
    /* 1. GLOBAL COLOR OVERRIDES (Bye Bye Red!) */
    :root {
        --primary-color: #4A90E2;
    }

    /* 2. SLIDER FIX: Force Blue track and handle */
    div[data-baseweb="slider"] > div:first-child {
        background: linear-gradient(to right, rgb(74, 144, 226) 0%, rgb(74, 144, 226) var(--slider-value), rgb(226, 232, 240) var(--slider-value), rgb(226, 232, 240) 100%) !important;
    }
    div[role="slider"] {
        background-color: #4A90E2 !important;
        border: 2px solid #FFFFFF !important;
    }
    div[data-testid="stThumbValue"] {
        color: #4A90E2 !important;
    }

    /* 3. CHECKBOX FIX: Force Blue background when checked */
    div[data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {
        background-color: #4A90E2 !important;
        border-color: #4A90E2 !important;
    }

    /* 4. BUTTON FIX: Mock Blue solid/gradient */
    div.stButton > button {
        background-color: #4A90E2 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #357ABD !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }

    /* 5. SIDEBAR & CARD POLISH */
    [data-testid="stSidebar"] { background-color: #F8FAFC !important; }
    .stProgress > div > div > div > div { background-color: #4A90E2 !important; }
    
    .rank-badge { background-color: #4A90E2; color: white; padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8em; }
    .skill-tag { background-color: #DCFCE7; color: #166534; padding: 4px 10px; border-radius: 20px; font-size: 0.75em; border: 1px solid #BBF7D0; }
    .gap-tag { background-color: #F1F5F9; color: #64748B; padding: 4px 10px; border-radius: 20px; font-size: 0.75em; border: 1px solid #E2E8F0; }
    </style>
    """, unsafe_allow_html=True)

# --- MODAL EVALUATION ---
@st.dialog("Detailed Candidate Evaluation", width="large")
def show_candidate_modal(res, role_input):
    st.subheader(f"👤 Candidate Analysis: {res['id']}")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### 🤖 AI Rationale")
        prompt = f"Explain fit for {res['id']} in {role_input}. Skills: {res['matched']}."
        st.write(ollama.generate(model='gemma3:4b', prompt=prompt)['response'])
    with c2:
        st.metric("Match Score", f"{round((res['match_count']/res['total_count'])*100)}%")
        for s in res['all_reqs']:
            st.write(f"{'✅' if s in res['matched'] else '❌'} {s}")

# --- DATA LAYER ---
def get_top_candidates(job_title, limit=3):
    driver = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "12345678"))
    with driver.session() as session:
        query = """
        MATCH (o:Occupation) WHERE o.name =~ $job_name
        MATCH (o)-[:REQUIRES]->(s:Skill)
        OPTIONAL MATCH (s)<-[:HAS_SKILL]-(c:Candidate)
        WITH o, c, collect(DISTINCT s.name) AS matched
        MATCH (o)-[:REQUIRES]->(all_s:Skill)
        WITH o, c, matched, collect(DISTINCT all_s.name) AS all_reqs
        WHERE c IS NOT NULL
        RETURN coalesce(toString(c.id), last(split(elementId(c), ':'))) AS id,
               size(matched) AS match_count, size(all_reqs) AS total_count,
               matched, all_reqs
        ORDER BY match_count DESC LIMIT $limit
        """
        return session.run(query, job_name=f"(?i).*{job_title}.*", limit=limit).data()

# --- SIDEBAR (MOCK MATCH) ---
with st.sidebar:
    st.markdown("### ⚙️ Search Configuration")
    with st.container(border=True):
        num_candidates = st.slider("Number of candidates", 1, 10, 5)
        min_score = st.slider("Minimum Match Score", 0.50, 1.00, 0.75)
        role_input = st.text_input("Job Description / Query", value="Senior Python Developer")
        
        find_btn = st.button("🔍 Find & Explain Best Candidates", use_container_width=True)
        
        st.info("ℹ️ Results are optimized for UI preview.")
        st.checkbox("Knowledge Graph Filtering", value=True)
        st.checkbox("Vector Similarity Search", value=True)
        st.markdown("<p style='font-size: 0.8em; color: #64748B; margin-top:10px;'>💾 Save search  |  More ❯</p>", unsafe_allow_html=True)

# --- MAIN DASHBOARD ---
if find_btn or role_input:
    results = get_top_candidates(role_input, limit=num_candidates)
    
    st.title("🤖 Intelligent Recruitment Assistant")
    
    # Bar Chart (Refined Colors)
    st.subheader("📊 Candidate Readiness Overview")
    ids = [f"ID: {r['id']}" for r in results]
    matches = [r['match_count'] for r in results]
    gaps = [r['total_count'] - r['match_count'] for r in results]

    fig = go.Figure(data=[
        go.Bar(name='Matched', x=ids, y=matches, marker_color='#22C55E'),
        go.Bar(name='Gap', x=ids, y=gaps, marker_color='#EF4444')
    ])
    fig.update_layout(barmode='stack', height=250, margin=dict(t=10, b=10, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Ranked Grid
    st.subheader("🏆 Ranked Candidates")
    cols = st.columns(3)
    for i, res in enumerate(results):
        with cols[i % 3]:
            match_pct = round((res['match_count'] / res['total_count']) * 100)
            with st.container(border=True):
                st.markdown(f"<div><span class='rank-badge'>Rank {i+1}</span> <b>CAND_{res['id']}</b> <span style='float:right; color:#64748B;'>{match_pct}%</span></div>", unsafe_allow_html=True)
                st.progress(res['match_count'] / res['total_count'])
                
                st.markdown("**🛡️ AI Match Rationale**")
                prompt = f"1-sentence fit summary for {res['id']} as {role_input}."
                summary = ollama.generate(model='gemma3:4b', prompt=prompt, options={"num_predict": 60, "stop":["\n"]})['response']
                st.markdown(f"<p style='font-size:0.85em; color:#334155;'>{summary.strip()}</p>", unsafe_allow_html=True)

                st.markdown("".join([f"<span class='skill-tag'>● {s}</span>" for s in res['matched'][:3]]), unsafe_allow_html=True)
                
                if st.button(f"View Profile", key=f"v_{res['id']}"):
                    show_candidate_modal(res, role_input)