import streamlit as st
import ollama
from neo4j import GraphDatabase
import plotly.graph_objects as go
import re

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Intelligent Recruitment Assistant", layout="wide")

st.markdown("""
    <style>
    /* 1. SIDEBAR CLEANUP (Transparent blocks) */
    [data-testid="stSidebar"] { background-color: #F8FAFC !important; }
    div[data-baseweb="slider"] > div:first-child { background-color: transparent !important; }
    div[data-baseweb="slider"] [role="presentation"] { background-color: #E2E8F0 !important; height: 6px !important; }
    div[role="slider"] { background-color: white !important; border: 2px solid #4A90E2 !important; height: 18px !important; width: 18px !important; }

    /* 2. BUTTONS & CARDS */
    div.stButton > button { background-color: #4A90E2 !important; color: white !important; border-radius: 6px !important; width: 100% !important; font-weight: 600 !important; }
    .rank-badge { background-color: #4A90E2; color: white; padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8em; }
    .skill-tag { background-color: #DCFCE7; color: #166534; padding: 4px 10px; border-radius: 20px; font-size: 0.75em; border: 1px solid #BBF7D0; margin-right: 5px; display: inline-block; margin-bottom: 5px; }
    
    /* Loading Spinner Color Fix */
    .stSpinner > div > div { border-top-color: #4A90E2 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- DIALOG (MODAL WINDOW) with PROGRESS INDICATOR ---
@st.dialog("Detailed Candidate Evaluation", width="large")
def show_candidate_modal(res, role_input):
    st.subheader(f"👤 Candidate Analysis: {res['id']}")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("#### 🤖 AI Rationale")
        # Added Spinner here for progress feedback
        with st.spinner("🧠 Gemma is analyzing skill match and generating fit rationale..."):
            prompt = f"Explain fit for {res['id']} in {role_input}. Skills: {res['matched']}."
            try:
                response = ollama.generate(model='gemma3:4b', prompt=prompt)['response']
                st.write(response)
            except Exception as e:
                st.error("AI Analysis failed. Please check Ollama connection.")
    
    with c2:
        match_pct = round((res['match_count']/res['total_count'])*100)
        st.metric("Match Score", f"{match_pct}%")
        
        st.markdown("**Skill Gap Analysis**")
        # Visual breakdown of skills
        for s in res['all_reqs'][:15]:
            icon = "✅" if s in res['matched'] else "❌"
            st.write(f"{icon} {s}")

# --- RADAR CHART GENERATOR ---
def create_comparison_radar(results):
    fig = go.Figure()
    all_req_skills = sorted(list(set([s for r in results for s in r['all_reqs']])))[:10]
    
    fig.add_trace(go.Scatterpolar(
        r=[100] * len(all_req_skills) + [100], theta=all_req_skills + [all_req_skills[0]],
        fill='toself', name='Benchmark', fillcolor='rgba(226, 232, 240, 0.3)',
        line=dict(color='#94A3B8', width=1, dash='dot')
    ))

    colors = ['#4A90E2', '#22C55E', '#F59E0B', '#8B5CF6', '#EC4899'] 
    for i, res in enumerate(results[:5]):
        r_values = [100 if s in res['matched'] else 15 for s in all_req_skills]
        r_values.append(r_values[0])
        fig.add_trace(go.Scatterpolar(
            r=r_values, theta=all_req_skills + [all_req_skills[0]],
            name=f"CAND_{res['id']}", line=dict(color=colors[i % len(colors)], width=2.5)
        ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=False), gridshape='linear'), 
                      showlegend=True, height=500, margin=dict(t=50, b=50, l=50, r=50))
    return fig

# --- DATA LAYER ---
def get_top_candidates(job_title, limit=5, threshold=0.5):
    driver = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "12345678"))
    with driver.session() as session:
        query = """
        MATCH (o:Occupation) WHERE o.name =~ $job_name
        MATCH (o)-[:REQUIRES]->(s:Skill)
        OPTIONAL MATCH (s)<-[:HAS_SKILL]-(c:Candidate)
        WITH o, c, collect(DISTINCT s.name) AS matched_skills
        MATCH (o)-[:REQUIRES]->(all_s:Skill)
        WITH o, c, matched_skills, collect(DISTINCT all_s.name) AS all_reqs
        WHERE c IS NOT NULL
        WITH c, matched_skills, all_reqs, size(matched_skills) as m_count, size(all_reqs) as t_count
        WHERE (toFloat(m_count)/t_count) >= $threshold
        RETURN coalesce(toString(c.id), last(split(elementId(c), ':'))) AS id,
               m_count AS match_count, t_count AS total_count, matched_skills AS matched, all_reqs
        ORDER BY match_count DESC LIMIT $limit
        """
        return session.run(query, job_name=f"(?i).*{job_title}.*", limit=limit, threshold=threshold).data()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Search Configuration")
    role_query = st.text_input("Role Query", value="software analyst")
    max_results = st.slider("Max Results", 1, 15, 5)
    min_threshold = st.slider("Min Match Threshold", 0.0, 1.0, 0.40)
    find_btn = st.button("🔍 Analyze Talent Pool")
    st.checkbox("Knowledge Graph Filtering", value=True)
    st.checkbox("Vector Similarity Search", value=True)

# --- MAIN DASHBOARD ---
if find_btn or role_query:
    # Top-level loading bar for the whole query
    with st.status("Searching Knowledge Graph...", expanded=False) as status:
        results = get_top_candidates(role_query, limit=max_results, threshold=min_threshold)
        status.update(label="Analysis Complete!", state="complete", expanded=False)
    
    if results:
        st.title("🤖 Intelligent Recruitment Assistant")
        st.subheader("🕸️ Talent Footprint Comparison")
        st.plotly_chart(create_comparison_radar(results), use_container_width=True)

        st.divider()
        st.subheader("🏆 Ranked Candidates")
        
        cols = st.columns(3)
        for i, res in enumerate(results):
            with cols[i % 3]:
                match_pct = round((res['match_count'] / res['total_count']) * 100)
                with st.container(border=True):
                    st.markdown(f"<div><span class='rank-badge'>Rank {i+1}</span> <b>CAND_{res['id']}</b> <span style='float:right;'>{match_pct}%</span></div>", unsafe_allow_html=True)
                    st.progress(res['match_count'] / res['total_count'])
                    
                    st.markdown("**🛡️ AI Match Rationale**")
                    # Quick one-liner for the card
                    summary = ollama.generate(model='gemma3:4b', prompt=f"1-sentence summary for {res['id']} as {role_query}.", options={"num_predict": 40})['response']
                    clean_summary = re.sub(r'[*#_>]', '', summary).strip()
                    st.markdown(f"<p style='font-size:0.85em; color:#334155; height:50px;'>{clean_summary}</p>", unsafe_allow_html=True)

                    st.markdown("".join([f"<span class='skill-tag'>● {s}</span>" for s in res['matched'][:4]]), unsafe_allow_html=True)
                    
                    if st.button("View Full Profile", key=f"btn_{res['id']}_{i}"):
                        show_candidate_modal(res, role_query)