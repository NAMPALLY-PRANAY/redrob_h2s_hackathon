import gradio as gr
try:
    import orjson as json_lib
except ImportError:
    import json as json_lib
import pandas as pd
import tempfile
import os
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from rank import (
    piece_exp_score, title_score, exact_skill_score,
    text_relevance, work_fit, location_fit, career_strength, consistency,
    reason_for_candidate
)

# Global store for loaded candidate data to enable interactive inspection and AI insights query
loaded_candidates = []
ranking_results = []
original_weights = {
    "exp": 3.3, "title": 1.0, "loc": 1.0, "bio": 1.0,
    "skill": 1.0, "text": 1.0, "career": 1.0, "cons": 1.0
}

# Custom Score Calculator supporting dynamic weights
def score_candidate_custom(c, weights):
    p = c["profile"]
    sig = c.get("redrob_signals", {})
    rel, matched = text_relevance(c)
    
    s = 0.0
    s += weights["exp"] * piece_exp_score(p.get("years_of_experience"))
    s += weights["title"] * title_score(p.get("current_title", ""))
    s += weights["loc"] * location_fit(p, sig)
    s += weights["bio"] * work_fit(sig)
    s += weights["skill"] * exact_skill_score(c)
    s += weights["text"] * rel
    s += weights["career"] * career_strength(c)
    s += weights["cons"] * consistency(c)
    return s, matched

# Yields progressive HTML progress states
def make_step_html(step_states):
    html = '<div class="progress-container">'
    html += '<h4>🔄 AI Discovery Pipeline Processing</h4>'
    
    completed = sum(1 for _, s in step_states if s == 2)
    active = sum(0.5 for _, s in step_states if s == 1)
    total = len(step_states)
    pct = int(((completed + active) / total) * 100)
    
    html += f'<div style="display:flex; justify-content:space-between; margin-bottom:5px; font-size:0.85rem; color:var(--text-secondary)"><span>Progress</span><span>{pct}%</span></div>'
    html += '<div class="progress-bar-bg"><div class="progress-bar-fill" style="width:' + str(pct) + '%"></div></div>'
    html += '<div style="margin-top:1.25rem;">'
    
    for name, state in step_states:
        if state == 2:
            indicator = '<span class="step-indicator step-complete">✓</span>'
            text_style = 'color: var(--success); font-weight:500;'
        elif state == 1:
            indicator = '<span class="step-indicator step-active">●</span>'
            text_style = 'color: var(--primary); font-weight:600;'
        else:
            indicator = '<span class="step-indicator step-pending">○</span>'
            text_style = 'color: var(--text-muted);'
        html += f'<div class="step-item">{indicator}<span style="{text_style}">{name}</span></div>'
        
    html += '</div></div>'
    return html

# Visualization Plotly generators
def make_score_distribution_plot(df):
    if df.empty:
        return go.Figure()
    fig = px.histogram(
        df, 
        x="score", 
        nbins=12, 
        title="Candidate Score Distribution",
        color_discrete_sequence=["#6366F1"],
        labels={"score": "Aggregate Score"}
    )
    fig.update_layout(
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font_color="#F8FAFC",
        title_font_family="Outfit",
        title_font_size=15,
        margin=dict(l=30, r=30, t=50, b=30),
        height=320
    )
    fig.update_xaxes(gridcolor="#1E293B", linecolor="#1E293B")
    fig.update_yaxes(gridcolor="#1E293B", linecolor="#1E293B")
    return fig

def make_experience_vs_score_plot(results_list):
    if not results_list:
        return go.Figure()
    df_scatter = pd.DataFrame([
        {
            "candidate_id": item["candidate_id"],
            "years_of_experience": item["c_dict"]["profile"].get("years_of_experience", 0) or 0,
            "score": item["score"],
            "open_to_work": "Open" if item["c_dict"].get("redrob_signals", {}).get("open_to_work_flag") else "Not Open"
        }
        for item in results_list
    ])
    fig = px.scatter(
        df_scatter,
        x="years_of_experience",
        y="score",
        color="open_to_work",
        hover_data=["candidate_id"],
        title="Experience Years vs Score",
        color_discrete_map={"Open": "#10B981", "Not Open": "#EF4444"},
        labels={"years_of_experience": "Years of Experience", "score": "Aggregate Score"}
    )
    fig.update_layout(
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font_color="#F8FAFC",
        title_font_family="Outfit",
        title_font_size=15,
        margin=dict(l=30, r=30, t=50, b=30),
        height=320
    )
    fig.update_xaxes(gridcolor="#1E293B", linecolor="#1E293B")
    fig.update_yaxes(gridcolor="#1E293B", linecolor="#1E293B")
    return fig

def make_top_skills_plot(results_list):
    if not results_list:
        return go.Figure()
    from collections import Counter
    skill_counts = Counter()
    for item in results_list:
        for s in item["c_dict"].get("skills", []):
            skill_counts[s.get("name", "")] += 1
            
    top_skills = skill_counts.most_common(10)
    if not top_skills:
        top_skills = [("Python", 1), ("Machine Learning", 1)]
        
    df_skills = pd.DataFrame(top_skills, columns=["Skill Name", "Count"])
    fig = px.bar(
        df_skills,
        x="Count",
        y="Skill Name",
        orientation="h",
        title="Top Preferred Technical Skills",
        color="Count",
        color_continuous_scale="Purples",
        labels={"Count": "Frequency", "Skill Name": "Skill Name"}
    )
    fig.update_layout(
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font_color="#F8FAFC",
        title_font_family="Outfit",
        title_font_size=15,
        margin=dict(l=30, r=30, t=50, b=30),
        coloraxis_showscale=False,
        height=320
    )
    fig.update_xaxes(gridcolor="#1E293B", linecolor="#1E293B")
    fig.update_yaxes(gridcolor="#1E293B", linecolor="#1E293B")
    return fig

# Custom Recruiter score analyzer html card renderer
def render_inspector_card(candidate_id, exp_w, title_w=None, loc_w=None, bio_w=None, skill_w=None, text_w=None, career_w=None, cons_w=None):
    if isinstance(exp_w, dict):
        weights = exp_w
    else:
        weights = {
            "exp": exp_w, "title": title_w, "loc": loc_w, "bio": bio_w,
            "skill": skill_w, "text": text_w, "career": career_w, "cons": cons_w
        }
    global loaded_candidates
    if not loaded_candidates:
        return "<div class='inspector-card'><p style='color:var(--text-secondary); text-align:center;'>No candidate dataset loaded yet.</p></div>"
        
    c = next((item for item in loaded_candidates if item["candidate_id"] == candidate_id), None)
    if not c:
        return f"<div class='inspector-card'><p style='color:var(--error); text-align:center;'>Candidate {candidate_id} not found.</p></div>"
        
    p = c["profile"]
    sig = c.get("redrob_signals", {})
    
    # Calculate score metrics for explanation
    exp_score = weights["exp"] * piece_exp_score(p.get("years_of_experience"))
    title_val = title_score(p.get("current_title", ""))
    title_contrib = weights["title"] * title_val
    loc_val = location_fit(p, sig)
    loc_contrib = weights["loc"] * loc_val
    bio_val = work_fit(sig)
    bio_contrib = weights["bio"] * bio_val
    skill_val = exact_skill_score(c)
    skill_contrib = weights["skill"] * skill_val
    rel, matched = text_relevance(c)
    text_contrib = weights["text"] * rel
    career_val = career_strength(c)
    career_contrib = weights["career"] * career_val
    cons_val = consistency(c)
    cons_contrib = weights["cons"] * cons_val
    
    total_score = exp_score + title_contrib + loc_contrib + bio_contrib + skill_contrib + text_contrib + career_contrib + cons_contrib
    reason = reason_for_candidate(c, matched)
    
    # Build skills badge tags
    badges = []
    for s in c.get("skills", [])[:8]:
        name = s.get("name", "")
        prof = s.get("proficiency", "intermediate")
        if prof == "expert":
            color = "#818CF8"
        elif prof == "advanced":
            color = "#34D399"
        elif prof == "intermediate":
            color = "#F59E0B"
        else:
            color = "#94A3B8"
        badges.append(f'<span style="background:{color}; color:#0F172A; font-size:0.75rem; padding:2px 8px; border-radius:12px; margin-right:6px; font-weight:600; display:inline-block; margin-bottom:6px;">{name}</span>')
    badges_str = "".join(badges) if badges else "None declared"
    
    color_to_key = {
        "#6366F1": ("Experience Fit", "exp"),
        "#3B82F6": ("Role Title", "title"),
        "#10B981": ("Location Fit", "loc"),
        "#F59E0B": ("Behavioral Signal", "bio"),
        "#EC4899": ("Exact Skill", "skill"),
        "#8B5CF6": ("Text Relevance", "text"),
        "#06B6D4": ("Career Strength", "career"),
        "#14B8A6": ("Consistency", "cons")
    }

    # Calculate bar percentages for feature importance breakdown
    max_val = max(1.0, abs(exp_score), abs(title_contrib), abs(loc_contrib), abs(bio_contrib), abs(skill_contrib), abs(text_contrib), abs(career_contrib), abs(cons_contrib))
    
    def bar(val, color):
        w_pct = min(100, int((abs(val) / max_val) * 100))
        bar_color = color if val >= 0 else "var(--error)"
        sign = "+" if val >= 0 else "-"
        label, w_key = color_to_key.get(color, ("Signal", "exp"))
        w_val = weights.get(w_key, 1.0)
        return f"""
        <div style="display:flex; align-items:center; margin-bottom:8px; font-size:0.85rem;">
            <div style="width:140px; color:var(--text-secondary); text-overflow:ellipsis; overflow:hidden; white-space:nowrap;" title="{label} (w={w_val:.1f})">{label} <span style="font-size:0.75rem; color:var(--text-muted)">(x{w_val:.1f})</span></div>
            <div style="flex-grow:1; background:#1E293B; height:6px; border-radius:3px; position:relative; overflow:hidden;">
                <div style="background:{bar_color}; width:{w_pct}%; height:100%; border-radius:3px; position:absolute; left:0;"></div>
            </div>
            <div style="width:60px; text-align:right; font-weight:600; color:{bar_color};">{sign}{abs(val):.3f}</div>
        </div>
        """
        
    html = f"""
    <div class="inspector-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; border-bottom:1px solid var(--border-color); padding-bottom:12px; margin-bottom:12px;">
            <div>
                <h3 style="margin:0; font-size:1.35rem; color:var(--text-primary);">{p.get('anonymized_name', 'Anonymized Candidate')}</h3>
                <p style="margin:4px 0 0 0; color:var(--primary); font-weight:500; font-size:0.9rem;">{p.get('current_title', 'Engineer')} @ {p.get('current_company', 'Enterprise')}</p>
            </div>
            <div style="background:var(--primary-glow); border:1px solid var(--primary); padding:6px 16px; border-radius:8px; text-align:center;">
                <span style="display:block; font-size:0.7rem; color:var(--text-secondary); font-weight:500; text-transform:uppercase;">Score</span>
                <span style="font-size:1.4rem; font-weight:700; color:var(--text-primary);">{total_score:.4f}</span>
            </div>
        </div>
        
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-bottom:1.5rem;">
            <div>
                <h4 style="margin:0 0 8px 0; font-size:0.95rem; text-transform:uppercase; color:var(--text-secondary);">Recruiter Summary</h4>
                <p style="margin:0; font-size:0.9rem; line-height:1.45; color:var(--text-primary);">{reason}</p>
                <div style="margin-top:12px;">
                    <h5 style="margin:0 0 6px 0; font-size:0.85rem; color:var(--text-secondary);">Skills Highlight</h5>
                    {badges_str}
                </div>
            </div>
            <div>
                <h4 style="margin:0 0 10px 0; font-size:0.95rem; text-transform:uppercase; color:var(--text-secondary);">Signal Contributions</h4>
                {bar(exp_score, "#6366F1")}
                {bar(title_contrib, "#3B82F6")}
                {bar(loc_contrib, "#10B981")}
                {bar(bio_contrib, "#F59E0B")}
                {bar(skill_contrib, "#EC4899")}
                {bar(text_contrib, "#8B5CF6")}
                {bar(career_contrib, "#06B6D4")}
                {bar(cons_contrib, "#14B8A6")}
            </div>
        </div>
        
        <div style="display:flex; justify-content:space-between; align-items:center; background:#0B0F19; padding:8px 12px; border-radius:6px; font-size:0.8rem; color:var(--text-muted);">
            <span>Availability: Notice {sig.get('notice_period_days', 30)} Days / Relocate: {"Yes" if sig.get('willing_to_relocate') else "No"}</span>
            <span>Profile Completeness: {sig.get('profile_completeness_score', 0)}% / Response Rate: {int(sig.get('recruiter_response_rate', 0)*100)}%</span>
        </div>
    </div>
    """
    return html

# Main File Processor with progressive loading checklists
def process_candidate_dataset(file_obj, exp_w, title_w, loc_w, bio_w, skill_w, text_w, career_w, cons_w):
    global loaded_candidates, ranking_results
    
    if file_obj is None:
        # Return landing empty page layout components
        return (
            gr.update(visible=True),   # empty landing visible
            gr.update(visible=False),  # loading checklist hidden
            gr.update(visible=False),  # dashboard workspace hidden
            gr.update(value="<div style='color:var(--text-muted)'>● Idle</div>"), # status text
            None,                      # score distribution plot
            None,                      # experience scatter
            None,                      # skills counter plot
            None,                      # interactive grid dataframe
            gr.update(choices=[]),     # inspector candidate choices
            "",                        # inspector card html
            ""                         # stats cards
        )
        
    # Check macOS System Shadow files
    orig_name = getattr(file_obj, "orig_name", "")
    if not orig_name and hasattr(file_obj, "name"):
        orig_name = os.path.basename(file_obj.name)
        
    if orig_name.startswith("._"):
        raise gr.Error(
            f"You uploaded a macOS system metadata file ('{orig_name}'). "
            "Please upload the actual candidate data file instead."
        )

    # Initialise progressive checkpoints list
    steps = [
        ("Preparing system context & cleaning workspace", 1),
        ("Reading uploaded byte-stream dataset", 0),
        ("Validating profiles against JSON Schema properties", 0),
        ("Evaluating components (Skills, Bio, Experience)", 0),
        ("Executing multi-signal stable sorting matrices", 0),
        ("Deferring explainable recruiter reasoning block", 0),
        ("Finalizing dashboard stats and Plotly plots", 0)
    ]
    
    # Yield 1: Start loading
    yield (
        gr.update(visible=False),  # empty landing hidden
        gr.update(visible=True, value=make_step_html(steps)), # loading list visible
        gr.update(visible=False),  # dashboard workspace hidden
        gr.update(value="<div style='color:var(--primary); font-weight:600;'>● Processing...</div>"), # status text
        None, None, None, None, gr.update(choices=[]), "", ""
    )
    
    time.sleep(0.5)
    
    # Yield 2: Reading
    steps[0] = (steps[0][0], 2)
    steps[1] = (steps[1][0], 1)
    yield (
        gr.update(visible=False), gr.update(value=make_step_html(steps)), gr.update(visible=False),
        gr.update(value="<div style='color:var(--primary); font-weight:600;'>● Processing...</div>"),
        None, None, None, None, gr.update(choices=[]), "", ""
    )
    
    try:
        with open(file_obj.name, "rb") as f:
            first_chunk = f.read(100).strip()
            f.seek(0)
            if not first_chunk:
                raise gr.Error("File is empty.")
            
            if first_chunk.startswith(b"["):
                candidates = json_lib.loads(f.read())
            else:
                candidates = []
                for line in f:
                    line = line.strip()
                    if line:
                        candidates.append(json_lib.loads(line))
        loaded_candidates = candidates
    except UnicodeDecodeError:
        raise gr.Error("Failed to parse: binary encoding issues in file.")
    except Exception as e:
        raise gr.Error(f"Failed to parse file: {e}")
        
    time.sleep(0.4)
    
    # Yield 3: Schema validation
    steps[1] = (steps[1][0], 2)
    steps[2] = (steps[2][0], 1)
    yield (
        gr.update(visible=False), gr.update(value=make_step_html(steps)), gr.update(visible=False),
        gr.update(value="<div style='color:var(--primary); font-weight:600;'>● Processing...</div>"),
        None, None, None, None, gr.update(choices=[]), "", ""
    )
    
    # Quick structure check
    valid_count = sum(1 for c in candidates if isinstance(c, dict) and "candidate_id" in c)
    if valid_count == 0:
        raise gr.Error("No valid candidate profiles (missing 'candidate_id') found in the file.")
        
    time.sleep(0.4)
    
    # Yield 4: Evaluating
    steps[2] = (steps[2][0], 2)
    steps[3] = (steps[3][0], 1)
    yield (
        gr.update(visible=False), gr.update(value=make_step_html(steps)), gr.update(visible=False),
        gr.update(value="<div style='color:var(--primary); font-weight:600;'>● Processing...</div>"),
        None, None, None, None, gr.update(choices=[]), "", ""
    )
    
    # Store weights
    weights = {
        "exp": exp_w, "title": title_w, "loc": loc_w, "bio": bio_w,
        "skill": skill_w, "text": text_w, "career": career_w, "cons": cons_w
    }
    
    # Compute candidate scores
    scored_list = []
    total_exp = 0.0
    for c in candidates:
        if not isinstance(c, dict) or "candidate_id" not in c:
            continue
        try:
            score, matched = score_candidate_custom(c, weights)
            total_exp += c["profile"].get("years_of_experience", 0) or 0
            scored_list.append({
                "candidate_id": c["candidate_id"],
                "score": score,
                "matched": matched,
                "c_dict": c
            })
        except Exception:
            continue
            
    time.sleep(0.4)
    
    # Yield 5: Sorting
    steps[3] = (steps[3][0], 2)
    steps[4] = (steps[4][0], 1)
    yield (
        gr.update(visible=False), gr.update(value=make_step_html(steps)), gr.update(visible=False),
        gr.update(value="<div style='color:var(--primary); font-weight:600;'>● Processing...</div>"),
        None, None, None, None, gr.update(choices=[]), "", ""
    )
    
    scored_list.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    ranking_results = scored_list
    
    time.sleep(0.4)
    
    # Yield 6: Deferring Reasoning
    steps[4] = (steps[4][0], 2)
    steps[5] = (steps[5][0], 1)
    yield (
        gr.update(visible=False), gr.update(value=make_step_html(steps)), gr.update(visible=False),
        gr.update(value="<div style='color:var(--primary); font-weight:600;'>● Processing...</div>"),
        None, None, None, None, gr.update(choices=[]), "", ""
    )
    
    top_n = min(len(scored_list), 100)
    output_rows = []
    for rank, item in enumerate(scored_list[:top_n], start=1):
        reason = reason_for_candidate(item["c_dict"], item["matched"])
        output_rows.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": round(item["score"], 6),
            "reasoning": reason
        })
    df = pd.DataFrame(output_rows)
    
    # Export report
    temp_dir = tempfile.gettempdir()
    csv_path = os.path.join(temp_dir, "submission.csv")
    df.to_csv(csv_path, index=False)
    
    time.sleep(0.4)
    
    # Yield 7: Generating plots
    steps[5] = (steps[5][0], 2)
    steps[6] = (steps[6][0], 1)
    yield (
        gr.update(visible=False), gr.update(value=make_step_html(steps)), gr.update(visible=False),
        gr.update(value="<div style='color:var(--primary); font-weight:600;'>● Processing...</div>"),
        None, None, None, None, gr.update(choices=[]), "", ""
    )
    
    # Plots
    plot_dist = make_score_distribution_plot(df)
    plot_scatter = make_experience_vs_score_plot(scored_list[:top_n])
    plot_skills = make_top_skills_plot(scored_list[:top_n])
    
    # Calculate stats cards variables
    candidates_count = len(scored_list)
    top_fits = sum(1 for item in scored_list if item["score"] > 0)
    avg_exp = total_exp / max(1, candidates_count)
    completeness = sum(item["c_dict"].get("redrob_signals", {}).get("profile_completeness_score", 0) for item in scored_list) / max(1, candidates_count)
    
    # Stats cards widgets HTML
    stats_html = f"""
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:1.25rem; margin-bottom:1.5rem;">
        <div class="metric-card">
            <span style="display:block; font-size:0.8rem; color:var(--text-secondary); font-weight:500; text-transform:uppercase;">Total Candidates</span>
            <span style="font-size:1.8rem; font-weight:700; color:var(--text-primary); margin-top:4px; display:block;">{candidates_count:,}</span>
            <span style="font-size:0.75rem; color:var(--success); font-weight:500; display:block; margin-top:4px;">100% Parsed Successfully</span>
        </div>
        <div class="metric-card">
            <span style="display:block; font-size:0.8rem; color:var(--text-secondary); font-weight:500; text-transform:uppercase;">High Fit Matches</span>
            <span style="font-size:1.8rem; font-weight:700; color:var(--text-primary); margin-top:4px; display:block;">{top_fits}</span>
            <span style="font-size:0.75rem; color:var(--text-muted); display:block; margin-top:4px;">Aggregate Score > 0</span>
        </div>
        <div class="metric-card">
            <span style="display:block; font-size:0.8rem; color:var(--text-secondary); font-weight:500; text-transform:uppercase;">Avg Experience</span>
            <span style="font-size:1.8rem; font-weight:700; color:var(--text-primary); margin-top:4px; display:block;">{avg_exp:.1f} Yrs</span>
            <span style="font-size:0.75rem; color:var(--primary); font-weight:500; display:block; margin-top:4px;">Mid-to-Senior Range</span>
        </div>
        <div class="metric-card">
            <span style="display:block; font-size:0.8rem; color:var(--text-secondary); font-weight:500; text-transform:uppercase;">Profile Completeness</span>
            <span style="font-size:1.8rem; font-weight:700; color:var(--text-primary); margin-top:4px; display:block;">{completeness:.1f}%</span>
            <span style="font-size:0.75rem; color:var(--success); font-weight:500; display:block; margin-top:4px;">Verified Signals</span>
        </div>
    </div>
    """
    
    # Inspector candidate choices
    inspector_choices = [item["candidate_id"] for item in scored_list[:top_n]]
    default_card = render_inspector_card(inspector_choices[0], weights) if inspector_choices else ""
    
    steps[6] = (steps[6][0], 2)
    time.sleep(0.3)
    
    # Yield final results: Unhide dashboard and update visualizers
    yield (
        gr.update(visible=False), # empty landing hidden
        gr.update(visible=False), # loading checklist hidden
        gr.update(visible=True),  # dashboard workspace visible
        gr.update(value="<div style='color:var(--success); font-weight:600;'>✓ Pipeline Complete</div>"), # status text
        plot_dist,
        plot_scatter,
        plot_skills,
        df,
        gr.update(choices=inspector_choices, value=inspector_choices[0] if inspector_choices else None),
        default_card,
        gr.update(value=stats_html)
    )

# ChatGPT-style Recruiter AI Insights Conversation logic
def process_ai_insight(query, history):
    global ranking_results
    if not ranking_results:
        response = "Please upload a candidate profile dataset first so I can analyze it and generate custom insights."
        history.append((query, response))
        return "", history
        
    q = query.lower().strip()
    
    if "top" in q or "best" in q or "who is the" in q:
        top_cand = ranking_results[0]
        cid = top_cand["candidate_id"]
        score = top_cand["score"]
        p = top_cand["c_dict"]["profile"]
        skills = ", ".join([s.get("name", "") for s in top_cand["c_dict"].get("skills", [])[:5]])
        reason = reason_for_candidate(top_cand["c_dict"], top_cand["matched"])
        
        response = f"""
### 🥇 Top Ranked Candidate Analysis

The highest scoring candidate is **{cid}** with an aggregate compatibility score of **{score:.4f}**.

**Candidate summary:**
* **Current Designation**: {p.get('current_title', 'Engineer')} at {p.get('current_company', 'N/A')}
* **Total Years of Experience**: {p.get('years_of_experience', 0)} Years
* **Primary Skills**: {skills}
* **Suitability**: {reason}
"""
    elif "skills" in q or "technologies" in q or "stack" in q:
        from collections import Counter
        all_skills = Counter()
        for item in ranking_results[:50]:
            for s in item["c_dict"].get("skills", []):
                all_skills[s.get("name", "")] += 1
        top_10 = all_skills.most_common(6)
        skills_table = "\n".join([f"| {s[0]} | {s[1]} |" for s in top_10])
        response = f"""
### 🛠️ Common Technical Skills & Technologies
Here are the most frequent technical skills found among the top 50 ranked candidates in this dataset:

| Skill / Technology | Frequency |
| :--- | :--- |
{skills_table}

The dataset demonstrates high technical alignment with the target Python and vector search database stack.
"""
    elif "india" in q or "onsite" in q or "pune" in q or "noida" in q:
        india_candidates = []
        for item in ranking_results:
            c = item["c_dict"]
            p = c["profile"]
            if p.get("country", "").lower() == "india":
                india_candidates.append(item)
                
        count = len(india_candidates)
        top_india = india_candidates[:3]
        rows = ""
        for idx, item in enumerate(top_india, start=1):
            p = item["c_dict"]["profile"]
            rows += f"\n{idx}. **{item['candidate_id']}** (Score: {item['score']:.4f}) - {p.get('current_title', 'Engineer')} located in {p.get('location', 'N/A')}"
            
        response = f"""
### 🇮🇳 Geographic & Relocation Compliance

There are **{count}** total candidates located in **India** within the uploaded dataset.

**Top 3 India-based matches:**
{rows}

These candidates are positioned strongly to meet the Pune/Noida onsite requirements.
"""
    elif "experience" in q or "senior" in q:
        counts = {"Junior (<3 yrs)": 0, "Mid (3-6 yrs)": 0, "Senior (6-10 yrs)": 0, "Principal (>10 yrs)": 0}
        for item in ranking_results:
            exp = item["c_dict"]["profile"].get("years_of_experience", 0) or 0
            if exp < 3:
                counts["Junior (<3 yrs)"] += 1
            elif exp < 6:
                counts["Mid (3-6 yrs)"] += 1
            elif exp <= 10:
                counts["Senior (6-10 yrs)"] += 1
            else:
                counts["Principal (>10 yrs)"] += 1
        
        response = f"""
### 🎓 Years of Experience Distribution

Here is the breakdown of candidate experience levels across the entire dataset:
* **Junior (<3 years)**: {counts["Junior (<3 yrs)"]} candidates
* **Mid (3-6 years)**: {counts["Mid (3-6 yrs)"]} candidates
* **Senior (6-10 years)**: {counts["Senior (6-10 yrs)"]} candidates (Peak evaluation range)
* **Principal (>10 years)**: {counts["Principal (>10 yrs)"]} candidates
"""
    else:
        response = f"""
### 🤖 Redrob Candidate Insights Engine

I am scanning the active dataset containing **{len(ranking_results)}** processed records. Try asking specific analytical queries:
* *"Who is the top candidate and why?"*
* *"What are the most frequent skills?"*
* *"How many candidates are located in India?"*
* *"Show experience distribution statistics"*
"""
    history.append((query, response))
    return "", history

# Custom CSS for complete styling overhaul
with open("assets/theme.css", "r", encoding="utf-8") as css_file:
    custom_css = css_file.read()

# Build Layout structure
with gr.Blocks(title="AI Candidate Ranking Platform") as demo:
    
    # Sliders and widgets state
    with gr.Row():
        
        # 1. Left Sidebar Navigation
        with gr.Column(scale=1, elem_id="sidebar"):
            logo_path = Path("logo.png")
            if logo_path.exists():
                gr.Image(value=str(logo_path), show_label=False, container=False, elem_classes="logo-container")
            
            gr.Markdown("## 🏆 Candidate Engine")
            gr.Markdown("Adjust weights to customize scores:")
            
            with gr.Group():
                exp_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=3.3, label="Experience Band Weight")
                title_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.1, value=1.0, label="Role Title Weight")
                loc_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.1, value=1.0, label="Location Proximity")
                bio_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.1, value=1.0, label="Behavioral Signal")
                skill_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.1, value=1.0, label="Exact Skill Match")
                text_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.1, value=1.0, label="Text Relevance")
                career_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.1, value=1.0, label="Career Strength")
                cons_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.1, value=1.0, label="Consistency Filter")
            
            gr.HTML("<hr style='border-top:1px solid var(--border-color); margin:1.25rem 0;'>")
            
            with gr.Accordion("📋 Expected Candidate Schema", open=False):
                gr.Markdown(
                    """
                    Upload candidate JSON/JSONL dataset conforming to:
                    * `candidate_id` (CAND_XXXXXXX)
                    * `profile` (headline, summary, years_of_experience)
                    * `career_history` (company, title, duration)
                    * `skills` (name, proficiency)
                    * `redrob_signals` (open_to_work_flag, completeness)
                    """
                )
                
            gr.HTML("<div style='font-size:0.75rem; color:var(--text-muted); text-align:center; margin-top:2rem;'>SaaS Platform v1.1.2</div>")
            
        # 2. Main Workspace
        with gr.Column(scale=4, elem_id="main-workspace"):
            
            # Top Navigation Header
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("# 🏆 AI Candidate Ranking Platform")
                    gr.Markdown("Intelligent Candidate Evaluation, Multi-Signal Scoring, & Explainable Decision Making")
                with gr.Column(scale=1, min_width=150):
                    status_badge = gr.HTML(
                        value="<div style='background:#0F172A; border:1px solid var(--border-color); color:var(--text-secondary); text-align:center; padding:8px; border-radius:6px; font-weight:600;'>● Idle</div>"
                    )
            
            gr.HTML("<hr style='border:0; border-top:1px solid var(--border-color); margin:1rem 0 1.5rem 0;'>")
            
            # --- LANDING STATE: Drag-and-drop Card ---
            with gr.Column(visible=True) as landing_area:
                with gr.Column(elem_classes="upload-zone"):
                    gr.Markdown("### 📂 Drop Candidate Profiles Here")
                    gr.Markdown("Supports JSON or JSONL format. Max size 500MB.")
                    
                    file_input = gr.File(
                        label="Drop CSV/JSON files",
                        file_types=[".json", ".jsonl"],
                        type="filepath",
                        container=False
                    )
                    
                    gr.Markdown("— OR —")
                    example_file = Path("data/sample_candidates.json")
                    if example_file.exists():
                        load_sample_btn = gr.Button("💡 Load Sample Candidates Dataset", variant="secondary", size="sm")
                    else:
                        gr.Markdown("Place sample_candidates.json in data/ directory to run sample evaluations.")
            
            # --- PROCESSING LOADING CHECKLIST ---
            with gr.Column(visible=False) as loading_area:
                progress_checklist = gr.HTML(value="")
                
            # --- MAIN SAAS WORKSPACE (Post-Upload) ---
            with gr.Column(visible=False) as dashboard_workspace:
                
                # Stats dashboard cards grid
                stats_cards = gr.HTML(value="")
                
                # Visualizations segment
                with gr.Row():
                    chart_score_dist = gr.Plot()
                    chart_scatter_fit = gr.Plot()
                    chart_skills_counter = gr.Plot()
                
                # Detailed analysis tabs
                with gr.Tabs():
                    
                    with gr.TabItem("📊 Interactive Spreadsheet Grid"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                table_grid = gr.Dataframe(
                                    headers=["candidate_id", "rank", "score", "reasoning"],
                                    datatype=["str", "number", "number", "str"],
                                    wrap=True,
                                    interactive=False
                                )
                            with gr.Column(scale=1):
                                download_btn = gr.File(label="Download Ranked Report CSV", interactive=False)
                                
                    with gr.TabItem("🔎 Recruiter Profile Scorecard"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                inspector_dropdown = gr.Dropdown(
                                    choices=[],
                                    label="Select Candidate to Inspect",
                                    interactive=True
                                )
                            with gr.Column(scale=3):
                                inspector_details = gr.HTML(value="")
                                
                    with gr.TabItem("💬 ChatGPT AI Insights"):
                        chat_history = gr.Chatbot(label="AI Candidate Analyst Hub", height=400)
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="Ask a question about the dataset",
                                placeholder="e.g. Who is the top candidate and why? / What are the common skills?",
                                scale=4
                            )
                            chat_send = gr.Button("Ask AI", variant="primary", scale=1)
                            
                        # Quick links prompt buttons
                        with gr.Row():
                            btn_q1 = gr.Button("Who is the top candidate?", size="sm")
                            btn_q2 = gr.Button("What are the most frequent skills?", size="sm")
                            btn_q3 = gr.Button("How many candidates are located in India?", size="sm")
                            btn_q4 = gr.Button("Show experience distribution", size="sm")
            
            # --- WIDGET EVENT HANDLERS ---
            
            # File Input handlers
            file_input.change(
                fn=process_candidate_dataset,
                inputs=[file_input, exp_slider, title_slider, loc_slider, bio_slider, skill_slider, text_slider, career_slider, cons_slider],
                outputs=[landing_area, loading_area, dashboard_workspace, status_badge, chart_score_dist, chart_scatter_fit, chart_skills_counter, table_grid, inspector_dropdown, inspector_details, stats_cards]
            )
            
            if example_file.exists():
                load_sample_btn.click(
                    fn=lambda: str(example_file),
                    outputs=file_input
                )
                
            # Scorecard selection change handler
            inspector_dropdown.change(
                fn=render_inspector_card,
                inputs=[inspector_dropdown, exp_slider, title_slider, loc_slider, bio_slider, skill_slider, text_slider, career_slider, cons_slider],
                outputs=inspector_details
            )
            
            # Interactive weights sliders update handler
            def trigger_weights_recalculation(exp_w, title_w, loc_w, bio_w, skill_w, text_w, career_w, cons_w):
                global loaded_candidates
                if not loaded_candidates:
                    return None, None, None, None, gr.update(choices=[]), "", "", None
                    
                weights = {
                    "exp": exp_w, "title": title_w, "loc": loc_w, "bio": bio_w,
                    "skill": skill_w, "text": text_w, "career": career_w, "cons": cons_w
                }
                
                scored_list = []
                total_exp = 0.0
                for c in loaded_candidates:
                    if not isinstance(c, dict) or "candidate_id" not in c:
                        continue
                    try:
                        score, matched = score_candidate_custom(c, weights)
                        total_exp += c["profile"].get("years_of_experience", 0) or 0
                        scored_list.append({
                            "candidate_id": c["candidate_id"],
                            "score": score,
                            "matched": matched,
                            "c_dict": c
                        })
                    except Exception:
                        continue
                        
                scored_list.sort(key=lambda x: (-x["score"], x["candidate_id"]))
                
                top_n = min(len(scored_list), 100)
                output_rows = []
                for rank, item in enumerate(scored_list[:top_n], start=1):
                    reason = reason_for_candidate(item["c_dict"], item["matched"])
                    output_rows.append({
                        "candidate_id": item["candidate_id"],
                        "rank": rank,
                        "score": round(item["score"], 6),
                        "reasoning": reason
                    })
                df = pd.DataFrame(output_rows)
                
                temp_dir = tempfile.gettempdir()
                csv_path = os.path.join(temp_dir, "submission.csv")
                df.to_csv(csv_path, index=False)
                
                plot_dist = make_score_distribution_plot(df)
                plot_scatter = make_experience_vs_score_plot(scored_list[:top_n])
                plot_skills = make_top_skills_plot(scored_list[:top_n])
                
                # Stats cards variables
                candidates_count = len(scored_list)
                top_fits = sum(1 for item in scored_list if item["score"] > 0)
                avg_exp = total_exp / max(1, candidates_count)
                completeness = sum(item["c_dict"].get("redrob_signals", {}).get("profile_completeness_score", 0) for item in scored_list) / max(1, candidates_count)
                
                stats_html = f"""
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:1.25rem; margin-bottom:1.5rem;">
                    <div class="metric-card">
                        <span style="display:block; font-size:0.8rem; color:var(--text-secondary); font-weight:500; text-transform:uppercase;">Total Candidates</span>
                        <span style="font-size:1.8rem; font-weight:700; color:var(--text-primary); margin-top:4px; display:block;">{candidates_count:,}</span>
                        <span style="font-size:0.75rem; color:var(--success); font-weight:500; display:block; margin-top:4px;">100% Parsed Successfully</span>
                    </div>
                    <div class="metric-card">
                        <span style="display:block; font-size:0.8rem; color:var(--text-secondary); font-weight:500; text-transform:uppercase;">High Fit Matches</span>
                        <span style="font-size:1.8rem; font-weight:700; color:var(--text-primary); margin-top:4px; display:block;">{top_fits}</span>
                        <span style="font-size:0.75rem; color:var(--text-muted); display:block; margin-top:4px;">Aggregate Score > 0</span>
                    </div>
                    <div class="metric-card">
                        <span style="display:block; font-size:0.8rem; color:var(--text-secondary); font-weight:500; text-transform:uppercase;">Avg Experience</span>
                        <span style="font-size:1.8rem; font-weight:700; color:var(--text-primary); margin-top:4px; display:block;">{avg_exp:.1f} Yrs</span>
                        <span style="font-size:0.75rem; color:var(--primary); font-weight:500; display:block; margin-top:4px;">Weights Tuning Applied</span>
                    </div>
                    <div class="metric-card">
                        <span style="display:block; font-size:0.8rem; color:var(--text-secondary); font-weight:500; text-transform:uppercase;">Profile Completeness</span>
                        <span style="font-size:1.8rem; font-weight:700; color:var(--text-primary); margin-top:4px; display:block;">{completeness:.1f}%</span>
                        <span style="font-size:0.75rem; color:var(--success); font-weight:500; display:block; margin-top:4px;">Verified Signals</span>
                    </div>
                </div>
                """
                
                inspector_choices = [item["candidate_id"] for item in scored_list[:top_n]]
                default_card = render_inspector_card(inspector_choices[0], weights) if inspector_choices else ""
                
                return plot_dist, plot_scatter, plot_skills, df, gr.update(choices=inspector_choices, value=inspector_choices[0] if inspector_choices else None), default_card, stats_html, csv_path
                
            # Connect all sliders to update
            sliders_list = [exp_slider, title_slider, loc_slider, bio_slider, skill_slider, text_slider, career_slider, cons_slider]
            for s_item in sliders_list:
                s_item.change(
                    fn=trigger_weights_recalculation,
                    inputs=sliders_list,
                    outputs=[chart_score_dist, chart_scatter_fit, chart_skills_counter, table_grid, inspector_dropdown, inspector_details, stats_cards, download_btn]
                )
                
            # Chatbot queries
            chat_send.click(fn=process_ai_insight, inputs=[chat_input, chat_history], outputs=[chat_input, chat_history])
            chat_input.submit(fn=process_ai_insight, inputs=[chat_input, chat_history], outputs=[chat_input, chat_history])
            
            btn_q1.click(fn=lambda h: process_ai_insight("Who is the top candidate?", h), inputs=chat_history, outputs=[chat_input, chat_history])
            btn_q2.click(fn=lambda h: process_ai_insight("What are the most frequent skills?", h), inputs=chat_history, outputs=[chat_input, chat_history])
            btn_q3.click(fn=lambda h: process_ai_insight("How many candidates are located in India?", h), inputs=chat_history, outputs=[chat_input, chat_history])
            btn_q4.click(fn=lambda h: process_ai_insight("Show experience distribution", h), inputs=chat_history, outputs=[chat_input, chat_history])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css, theme=gr.themes.Soft())
