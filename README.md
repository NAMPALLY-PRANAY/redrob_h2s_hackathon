# 🚀 X7F9A2 — Intelligent Candidate Discovery & Ranking Engine

### Redrob Data & AI Challenge 2026

An AI-powered, deterministic candidate discovery and ranking system designed to intelligently match talent against a **Senior AI Engineer** job description. It evaluates professional summaries, career history, technical skills, geographical locations, and behavioral signals to generate recruiter-friendly explanations and stable rankings.

<p align="center">
  <img src="logo.png" alt="Team X7F9A2 Logo" width="300" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" />
</p>

---

## 👥 Team Identity

* **Team Name**: X7F9A2
* **Solo Participant**: PRANAY NAMPALLY
* **Email**: npranay2006@gmail.com
* **Phone**: +91-9440971722
* **Hugging Face Sandbox**: [https://huggingface.co/spaces/npran24/redrob_h2s_hackathon](https://huggingface.co/spaces/npran24/redrob_h2s_hackathon)
* **GitHub Repository**: [https://github.com/NAMPALLY-PRANAY/redrob_h2s_hackathon](https://github.com/NAMPALLY-PRANAY/redrob_h2s_hackathon)

---

## 🎯 Problem Statement & Objective

Traditional Applicant Tracking Systems (ATS) rely heavily on exact keyword matching, which fails to capture candidates who express experience differently or penalize qualified profiles that don't load resumes with spam keywords.

The **X7F9A2 Engine** implements a deterministic multi-signal ranking framework that:
1. Matches semantic concepts across career history and skill matrices.
2. Models candidate availability, activity, and notice periods using behavioral signals.
3. Leverages a weighted scoring matrix to rank the top 100 candidates.
4. Generates concise, fact-based explanations for recruiter review.

---

## 🏗️ System Architecture

The following diagram illustrates the candidate processing and evaluation lifecycle:

```mermaid
graph TD
    A[Job Description Requirements] --> B[Scoring & Ranking Engine]
    C[Candidate Profiles JSONL] --> B
    D[Redrob Behavioral Signals] --> B
    
    subgraph Engine [X7F9A2 Core Scoring Framework]
        B --> F[Experience Fit Score]
        B --> G[Title Alignment Score]
        B --> H[Target Skill Matching]
        B --> I[Location & Relocation Fit]
        B --> J[Engagement & Verification]
        
        F & G & H & I & J --> K[Weighted Aggregate Score]
        K --> L[Consistency Filters & Penalty Checks]
    end
    
    L --> M[Stable Sorter]
    M --> N[Factual Reasoning Generator]
    N --> O[submission.csv Output]
```

---

## 📊 Scoring Component Details

The engine evaluates candidates across several dimensions to balance technical alignment, availability, and retention signals:

### 1. Experience Fit Score (30% weight)
Uses a piece-wise function that maps total years of experience, peaking in the **5 to 9 years** range (ideal for a Senior role) and penalizing profiles with too little (e.g. <3 years) or excess seniority.

### 2. Title & Role Alignment (15% weight)
Assigns positive weights for target titles (e.g., `Machine Learning Engineer`, `AI Engineer`, `Search/Ranking Engineer`) and strong negative penalties for unrelated domains (e.g., `Marketing`, `HR`, `Accountant`).

### 3. Skill & Keyword Matching (25% weight)
Extracts target technologies including Python, embeddings, vector databases (Faiss, Milvus, Qdrant, Pinecone), Sentence Transformers, learning to rank, MLOps tools, and LLM fine-tuning techniques.

### 4. Behavioral Signals (15% weight)
Leverages `redrob_signals` to boost candidates who are actively open to work, respond quickly to recruiters, have verified accounts, and exhibit high Git activity scores, while applying penalties for extremely long notice periods (e.g. >90 days).

### 5. Location & Work Proximity (10% weight)
Boosts candidates residing in or willing to relocate to key tech centers in India (Pune, Noida, Bengaluru, Mumbai, Delhi-NCR, Hyderabad), and matches preferred work modes.

### 6. Consistency Filters (5% weight)
Cross-references profile claims with career history details (e.g. checks for keyword stuffing or experience inflation) and penalizes inconsistencies.

---

## 🔄 Data Pipeline Flow

```mermaid
sequenceDiagram
    participant Recruiter as Recruiter / CLI
    participant Loader as Data Loader
    participant Engine as X7F9A2 Engine
    participant Validator as Official Validator
    
    Recruiter->>Loader: Runs rank.py (candidates.jsonl)
    Loader->>Engine: Stream records line-by-line (low RAM usage)
    Engine->>Engine: Extract profile structure & career history
    Engine->>Engine: Compute multi-signal scores
    Engine->>Engine: Apply penalization & consistency filters
    Engine->>Engine: Stable sort: Score DESC, candidate_id ASC
    Engine->>Engine: Generate concise, fact-based reasoning
    Engine->>Recruiter: Export submission.csv
    Recruiter->>Validator: Runs validate_submission.py
    Validator->>Recruiter: Format valid confirmation!
```

---

## 📁 Repository Structure

```text
redrob_h2s_hackathon/
│
├── README.md                      # Detailed project documentation & diagrams
├── submission_metadata.yaml       # Challenge portal metadata verification
├── requirements.txt               # App and environment dependencies
├── rank.py                        # Core deterministic ranking pipeline script
├── app.py                         # Gradio sandbox application for local & HF Spaces
├── validate_submission.py         # Official format validation script
├── submission.csv                 # Generated top-100 candidates submission file
├── logo.png                       # Team X7F9A2 logo
│
├── candidate_schema.json          # Candidate record JSON schema definition
├── sample_candidates.json         # Official candidate evaluation sample subset
│
├── job_description.docx           # Senior AI Engineer job description
├── redrob_signals_doc.docx        # Behavioral signals explanation
├── submission_spec.docx           # Challenge rules and requirements
└── sample_submission.csv          # Starter submission file template
```

---

## ⚡ Reproducibility & Execution

Follow these instructions to run the engine locally or launch the Gradio sandbox:

### Prerequisites
Make sure Python 3.10+ is installed on your machine.

### 1. Local Pipeline Execution
Run the core ranking script to regenerate the `submission.csv` file from candidate data:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
*(Note: It also works with gzip files like `candidates.jsonl.gz`)*

### 2. Format Validation
Verify that the output format strictly complies with the hackathon validation rules:
```bash
python validate_submission.py submission.csv
```

### 3. Interactive Gradio Sandbox
First, install the environment dependencies:
```bash
pip install -r requirements.txt
```
Next, launch the interactive UI:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:7860` to upload `sample_candidates.json` and explore the candidates, scores, and recruiter-friendly reasoning.

---

## 🔮 Future Improvements

1. **Hybrid Retrieval**: Combine BM25 semantic text scores with Dense Retrieval Embeddings.
2. **Cross-Encoder Reranking**: Utilize a lightweight cross-encoder model to capture nuanced experiences.
3. **Learning to Rank (LTR)**: Train a gradient boosted decision tree (GBDT) model calibrated directly with recruiter interaction logs.
4. **LLM Extraction**: Use fine-tuned models to extract structured parameters from unformatted job descriptions.
