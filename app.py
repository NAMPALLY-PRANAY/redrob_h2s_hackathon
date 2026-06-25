import gradio as gr
try:
    import orjson as json_lib
except ImportError:
    import json as json_lib
import pandas as pd
import tempfile
import os
from pathlib import Path
from rank import score_candidate, reason_for_candidate

def run_ranker(file_obj):
    if file_obj is None:
        return None, None
    
    # Read file contents and determine format
    try:
        # Check if it is a macOS metadata file
        orig_name = getattr(file_obj, "orig_name", "")
        if not orig_name and hasattr(file_obj, "name"):
            orig_name = os.path.basename(file_obj.name)
            
        if orig_name.startswith("._"):
            raise gr.Error(
                f"You uploaded a macOS system metadata file ('{orig_name}'). "
                "Please upload the actual candidate data file instead."
            )
            
        with open(file_obj.name, "rb") as f:
            # Check the first few bytes to determine format
            first_chunk = f.read(100).strip()
            f.seek(0)
            
            if not first_chunk:
                raise gr.Error("The uploaded file is empty.")
            
            if first_chunk.startswith(b"["):
                # Standard JSON array of candidates
                candidates = json_lib.loads(f.read())
            else:
                # JSONL format: read line-by-line using a generator
                candidates = []
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        candidates.append(json_lib.loads(line))
                    except Exception as je:
                        raise gr.Error(f"Line {i} is not valid JSON: {je}")
    except gr.Error as ge:
        raise ge
    except UnicodeDecodeError:
        raise gr.Error(
            "Failed to read the file because it contains binary data. "
            "Please ensure it is a plain text file in JSON or JSONL format (and not a compressed .zip, .gz, or system metadata file)."
        )
    except Exception as e:
        raise gr.Error(f"Failed to parse file: {e}. Please ensure it is valid JSON or JSONL format.")
    
    if not candidates:
        raise gr.Error("No candidates were found in the file.")
        
    # Evaluate candidates using rank.py core logic
    # OPTIMIZATION: Only calculate score first, do NOT generate expensive reason text for all candidates
    results = []
    for c in candidates:
        if not isinstance(c, dict) or "candidate_id" not in c:
            continue
        try:
            score, matched = score_candidate(c)
            results.append({
                "candidate_id": c["candidate_id"],
                "score": score,
                "matched": matched,
                "c_dict": c
            })
        except Exception as e:
            # Skip invalid candidate records gracefully
            continue
            
    if not results:
        raise gr.Error("No valid candidate profiles with 'candidate_id' could be processed.")
        
    # Sort: Primary sort by score (descending), secondary sort by candidate_id (ascending) for ties
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    # OPTIMIZATION: Only generate reasoning for the top 100 candidates (or length of results, if less)
    top_n = min(len(results), 100)
    output_rows = []
    for rank, item in enumerate(results[:top_n], start=1):
        reason = reason_for_candidate(item["c_dict"], item["matched"])
        output_rows.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": round(item["score"], 6),
            "reasoning": reason
        })
        
    df = pd.DataFrame(output_rows)
    
    # Save the full results to a temporary CSV file for user download
    temp_dir = tempfile.gettempdir()
    csv_path = os.path.join(temp_dir, "submission.csv")
    df.to_csv(csv_path, index=False)
    
    return df, csv_path

# Custom CSS for modern styling
custom_css = """
footer {visibility: hidden}
.logo-container img {
    max-height: 90px;
    margin: 0 auto;
}
.schema-box pre {
    background-color: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 4px;
    padding: 10px;
}
.header-box {
    margin-bottom: 20px;
}
"""

with gr.Blocks(title="AI Candidate Ranking Platform") as demo:
    with gr.Row(elem_classes="header-box"):
        with gr.Column(scale=1):
            logo_path = Path("logo.png")
            if logo_path.exists():
                gr.Image(value=str(logo_path), show_label=False, container=False, elem_classes="logo-container")
        with gr.Column(scale=5):
            gr.Markdown(
                """
                # 🏆 AI Candidate Ranking Platform
                ### Intelligent Candidate Evaluation & Ranking (Team X7F9A2)
                
                Welcome to the AI Candidate Ranking Platform, an interactive application for evaluating and ranking candidate profiles using a deterministic scoring engine.
                """
            )
            
    gr.HTML("<hr style='border: 0; border-top: 1px solid var(--border-color-primary); margin: 15rem 0;'>")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📂 1. Upload Candidates")
            file_input = gr.File(
                label="Select candidate dataset (.json or .jsonl)",
                file_types=[".json", ".jsonl"],
                type="filepath"
            )
            
            gr.Markdown("#### 💡 Example Dataset")
            example_file = Path("data/sample_candidates.json")
            if example_file.exists():
                gr.Examples(
                    examples=[[str(example_file)]],
                    inputs=file_input,
                    label="Click to load official sample dataset"
                )
            else:
                gr.Markdown("No sample candidate dataset found.")
            
            run_btn = gr.Button("🔮 Process & Rank Candidates", variant="primary")
            
            with gr.Accordion("📋 Expected Candidate Schema", open=False):
                gr.Markdown(
                    """
                    Uploaded candidate files must comply with the official schema `data/candidate_schema.json`.
                    
                    **Required Top-Level Fields:**
                    * `candidate_id`: String (Format: `CAND_XXXXXXX`)
                    * `profile`: Object (anonymized_name, headline, summary, location, country, years_of_experience, current_title, current_company)
                    * `career_history`: Array of past roles (company, title, duration_months, description)
                    * `skills`: Array of skills (name, proficiency, endorsements)
                    * `redrob_signals`: Object of behavioral engagement factors
                    
                    **Example candidate JSON structure:**
                    ```json
                    {
                      "candidate_id": "CAND_0000001",
                      "profile": {
                        "anonymized_name": "Ira Vora",
                        "headline": "Backend Engineer | SQL, Spark, Cloud",
                        "summary": "Software / data professional with 6.9 years of experience...",
                        "location": "Toronto",
                        "country": "Canada",
                        "years_of_experience": 6.9,
                        "current_title": "Backend Engineer",
                        "current_company": "Mindtree"
                      },
                      "career_history": [...],
                      "skills": [...],
                      "redrob_signals": {
                        "profile_completeness_score": 86.9,
                        "open_to_work_flag": true,
                        "recruiter_response_rate": 0.34,
                        "notice_period_days": 60,
                        "willing_to_relocate": false,
                        "github_activity_score": 9.2,
                        "interview_completion_rate": 0.71
                      }
                    }
                    ```
                    """
                )
            
        with gr.Column(scale=3):
            gr.Markdown("### 📊 2. Ranked Candidates Excel View")
            download_output = gr.File(label="Download Ranked submission.csv Report", interactive=False)
            table_output = gr.Dataframe(
                headers=["candidate_id", "rank", "score", "reasoning"],
                datatype=["str", "number", "number", "str"],
                label="Interactive Candidates List View (Top 100)",
                wrap=True,
                interactive=False
            )
            
    # Set up event handlers
    run_btn.click(
        fn=run_ranker,
        inputs=file_input,
        outputs=[table_output, download_output]
    )
    
    file_input.change(
        fn=run_ranker,
        inputs=file_input,
        outputs=[table_output, download_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css, theme=gr.themes.Soft())
