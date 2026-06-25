import gradio as gr
import json
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
        with open(file_obj.name, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        if not content:
            raise gr.Error("The uploaded file is empty.")
            
        if content.startswith("["):
            # Standard JSON array of candidates
            candidates = json.loads(content)
        else:
            # JSONL format (one candidate JSON per line)
            candidates = []
            for i, line in enumerate(content.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError as je:
                    raise gr.Error(f"Line {i} is not valid JSON: {je}")
    except gr.Error as ge:
        raise ge
    except Exception as e:
        raise gr.Error(f"Failed to parse file: {e}. Please ensure it is valid JSON or JSONL format.")
    
    if not candidates:
        raise gr.Error("No candidates were found in the file.")
        
    # Evaluate candidates using rank.py core logic
    results = []
    for c in candidates:
        if not isinstance(c, dict) or "candidate_id" not in c:
            continue
        try:
            score, matched = score_candidate(c)
            reason = reason_for_candidate(c, matched)
            results.append({
                "candidate_id": c["candidate_id"],
                "score": score,
                "reasoning": reason
            })
        except Exception as e:
            # Skip invalid candidate records gracefully
            continue
            
    if not results:
        raise gr.Error("No valid candidate profiles with 'candidate_id' could be processed.")
        
    # Sort: Primary sort by score (descending), secondary sort by candidate_id (ascending) for ties
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    # Format the final ranked output rows
    output_rows = []
    for rank, item in enumerate(results, start=1):
        output_rows.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": round(item["score"], 6),
            "reasoning": item["reasoning"]
        })
        
    df = pd.DataFrame(output_rows)
    
    # Save the full results to a temporary CSV file for user download
    temp_dir = tempfile.gettempdir()
    csv_path = os.path.join(temp_dir, "submission.csv")
    df.to_csv(csv_path, index=False)
    
    # Show up to 100 rows in the preview (or the full list if it is smaller)
    preview_df = df.head(100)
    
    return preview_df, csv_path

# Custom CSS for modern styling
custom_css = """
footer {visibility: hidden}
.logo-container img {
    max-height: 100px;
    margin: 0 auto;
}
.header-center {
    text-align: center;
    margin-bottom: 2rem;
}
.card {
    border-radius: 8px;
    padding: 1.5rem;
    background-color: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
}
"""

with gr.Blocks(title="X7F9A2 — Intelligent Candidate Discovery Engine", css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            logo_path = Path("logo.png")
            if logo_path.exists():
                gr.Image(value=str(logo_path), show_label=False, container=False, elem_classes="logo-container")
        with gr.Column(scale=4):
            gr.Markdown(
                """
                # 🚀 X7F9A2 — Intelligent Candidate Discovery & Ranking Engine
                ### Redrob Data & AI Challenge 2026 Sandbox
                
                This interactive sandbox runs our proprietary deterministic ranking pipeline to match candidate profiles against the **Senior AI Engineer** job requirements. Upload a profile file to score candidates, view their rank, and download the formatted `submission.csv` report.
                """
            )
            
    gr.HTML("<hr style='border: 0; border-top: 1px solid var(--border-color-primary); margin: 1.5rem 0;'>")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📂 1. Upload Profiles")
            file_input = gr.File(
                label="Upload candidates.json or candidates.jsonl file",
                file_types=[".json", ".jsonl"],
                type="filepath"
            )
            
            gr.Markdown("#### 💡 Example Datasets")
            example_file = Path("sample_candidates.json")
            if example_file.exists():
                gr.Markdown(
                    f"You can download and use the official `{example_file.name}` dataset included in the workspace to test the engine."
                )
                gr.Examples(
                    examples=[[str(example_file)]],
                    inputs=file_input,
                    label="Click to load sample candidate file"
                )
            else:
                gr.Markdown("No example files found in root directory.")
                
            run_btn = gr.Button("🔮 Process & Rank Candidates", variant="primary")
            
        with gr.Column(scale=3):
            gr.Markdown("### 📊 2. Discovery Results")
            download_output = gr.File(label="Download Full Submission CSV File", interactive=False)
            table_output = gr.Dataframe(
                headers=["candidate_id", "rank", "score", "reasoning"],
                datatype=["str", "number", "number", "str"],
                label="Ranked Profiles Preview (Top 100)",
                wrap=True
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
    # Launch app
    demo.launch(server_name="0.0.0.0", server_port=7860)
