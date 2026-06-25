#!/usr/bin/env python3
"""
Redrob hackathon ranker.

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Supports .jsonl and .jsonl.gz. No network, CPU-only, deterministic.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

TODAY = date(2026, 6, 26)

TITLE_POS = [
    ("machine learning engineer", 1.5),
    ("ai engineer", 1.45),
    ("search engineer", 1.45),
    ("ranking engineer", 1.4),
    ("recommendation systems engineer", 1.35),
    ("recommendation engineer", 1.2),
    ("ml engineer", 1.25),
    ("applied scientist", 1.0),
    ("data scientist", 0.95),
    ("backend engineer", 0.75),
    ("data engineer", 0.85),
    ("software engineer", 0.75),
    ("platform engineer", 0.7),
]

TITLE_NEG = [
    ("marketing", -1.1),
    ("customer support", -1.2),
    ("operations manager", -1.0),
    ("accountant", -1.2),
    ("hr manager", -1.0),
    ("student", -1.3),
    ("intern", -1.2),
    ("sales", -1.0),
    ("teacher", -0.9),
    ("designer", -0.6),
]

CORE_PHRASES = {
    "embeddings": [
        "embeddings",
        "embedding drift",
        "sentence transformers",
        "sentence-transformers",
        "bge",
        "openai embeddings",
        "similarity search",
    ],
    "retrieval": [
        "retrieval",
        "candidate matching",
        "hybrid retrieval",
        "search",
        "vector search",
        "semantic search",
    ],
    "ranking": [
        "ranking",
        "ranker",
        "re-ranking",
        "reranking",
        "learning to rank",
        "ltr",
        "ndcg",
        "mrr",
        "map",
        "precision@10",
        "precision@5",
    ],
    "vector": [
        "vector database",
        "vector db",
        "pinecone",
        "weaviate",
        "qdrant",
        "milvus",
        "opensearch",
        "elasticsearch",
        "faiss",
        "pgvector",
    ],
    "eval": [
        "evaluation",
        "offline benchmark",
        "ab test",
        "a/b test",
        "feedback loop",
        "recruiter feedback",
        "metrics",
    ],
    "llm": ["llm", "fine-tuning", "lora", "qlora", "peft"],
    "hr": ["recruiting", "recruiter", "talent", "marketplace", "hr-tech", "candidate"],
}

FIELD_WEIGHTS = {"headline": 1.6, "summary": 1.0, "career": 1.3, "skills": 1.5, "education": 0.3}


def norm(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\+\#\.\-/ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_date(s: str | None):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        return None


def open_candidates(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def piece_exp_score(years: float | None) -> float:
    if years is None:
        return 0.0
    y = float(years)
    if y < 1:
        return -1.0
    if y < 3:
        return -0.4
    if y < 5:
        return 0.4 + 0.2 * (y - 3)  # 0.4..0.8
    if y <= 9:
        return 1.6 - 0.03 * (y - 7) ** 2  # peak around 7
    if y <= 12:
        return 1.0 - 0.07 * (y - 9)
    if y <= 15:
        return 0.5 - 0.07 * (y - 12)
    return -0.2


def title_score(title: str) -> float:
    t = norm(title)
    s = 0.0
    for k, w in TITLE_POS:
        if k in t:
            s += w
    for k, w in TITLE_NEG:
        if k in t:
            s += w
    return s


def exact_skill_score(c: Dict) -> float:
    skills = [norm(s.get("name", "")) for s in c.get("skills", [])]
    skillset = set(skills)
    important = {
        "python": 0.8,
        "embeddings": 1.1,
        "sentence transformers": 1.0,
        "sentence-transformers": 1.0,
        "faiss": 1.1,
        "milvus": 1.0,
        "qdrant": 1.0,
        "weaviate": 1.0,
        "pinecone": 1.0,
        "opensearch": 1.0,
        "elasticsearch": 1.0,
        "retrieval": 0.8,
        "information retrieval": 1.0,
        "ranking": 0.8,
        "ndcg": 0.7,
        "mrr": 0.7,
        "map": 0.5,
        "learning to rank": 0.9,
        "llm": 0.4,
        "fine-tuning": 0.4,
        "lora": 0.35,
        "qlora": 0.35,
        "peft": 0.35,
        "mlflow": 0.3,
        "hnsw": 0.5,
        "hybrid search": 1.0,
        "vector database": 1.0,
        "vector db": 1.0,
        "spark": 0.2,
        "airflow": 0.2,
        "kafka": 0.2,
    }
    s = 0.0
    for k, w in important.items():
        if k in skillset or any(k == x or k in x for x in skillset):
            s += w
    return s


def text_relevance(c: Dict) -> Tuple[float, List[str]]:
    p = c["profile"]
    text_fields = {
        "headline": norm(p.get("headline", "")),
        "summary": norm(p.get("summary", "")),
        "career": norm(
            " ".join(
                (x.get("title", "") + " " + x.get("description", "") + " " + x.get("company", ""))
                for x in c.get("career_history", [])
            )
        ),
        "skills": norm(" ".join(s.get("name", "") for s in c.get("skills", []))),
        "education": norm(" ".join((x.get("field_of_study", "") + " " + x.get("degree", "")) for x in c.get("education", []))),
    }

    s = 0.0
    matched_terms: List[str] = []

    for group, phrases in CORE_PHRASES.items():
        group_count = 0.0
        for ph in phrases:
            for field, txt in text_fields.items():
                if ph in txt:
                    group_count += FIELD_WEIGHTS[field]

        if group == "embeddings":
            s += min(group_count, 4) * 1.1
        elif group == "retrieval":
            s += min(group_count, 4) * 1.0
        elif group == "ranking":
            s += min(group_count, 4) * 1.05
        elif group == "vector":
            s += min(group_count, 4) * 1.1
        elif group == "eval":
            s += min(group_count, 3) * 0.9
        elif group == "llm":
            s += min(group_count, 3) * 0.45
        elif group == "hr":
            s += min(group_count, 2) * 0.35

        if group_count > 0:
            matched_terms.append(group)

    if "embeddings" in matched_terms and ("retrieval" in matched_terms or "vector" in matched_terms):
        s += 1.2
    if "ranking" in matched_terms and "eval" in matched_terms:
        s += 0.8
    if "hr" in matched_terms and ("retrieval" in matched_terms or "ranking" in matched_terms):
        s += 0.4

    return s, matched_terms


def work_fit(sig: Dict) -> float:
    s = 0.0
    if sig.get("open_to_work_flag"):
        s += 1.3
    else:
        s -= 1.6

    lad = parse_date(sig.get("last_active_date", ""))
    if lad:
        days = (TODAY - lad).days
        if days <= 30:
            s += 0.8
        elif days <= 90:
            s += 0.55
        elif days <= 180:
            s += 0.2
        elif days <= 365:
            s -= 0.2
        else:
            s -= 0.6

    resp = sig.get("recruiter_response_rate")
    if isinstance(resp, (int, float)):
        s += (resp - 0.25) * 1.3

    comp = sig.get("profile_completeness_score")
    if isinstance(comp, (int, float)):
        s += (comp - 60) / 100 * 0.3

    if sig.get("verified_email"):
        s += 0.15
    if sig.get("verified_phone"):
        s += 0.15
    if sig.get("linkedin_connected"):
        s += 0.15

    gh = sig.get("github_activity_score", -1)
    if isinstance(gh, (int, float)) and gh >= 0:
        s += min(gh, 80) / 80 * 0.65

    noc = sig.get("notice_period_days")
    if isinstance(noc, (int, float)):
        if noc <= 30:
            s += 0.3
        elif noc <= 60:
            s += 0.18
        elif noc <= 90:
            s += 0.06
        elif noc <= 120:
            s -= 0.15
        else:
            s -= 0.35

    ipr = sig.get("interview_completion_rate")
    if isinstance(ipr, (int, float)):
        s += (ipr - 0.6) * 0.55

    return s


def location_fit(p: Dict, sig: Dict) -> float:
    country = norm(p.get("country", ""))
    loc = norm(p.get("location", ""))
    s = 0.0

    if country == "india":
        s += 1.15
        if any(
            city in loc
            for city in [
                "pune",
                "noida",
                "delhi",
                "gurgaon",
                "gurugram",
                "mumbai",
                "bengaluru",
                "bangalore",
                "hyderabad",
                "chennai",
                "kolkata",
                "coimbatore",
                "kochi",
                "trivandrum",
                "vizag",
                "visakhapatnam",
                "jaipur",
            ]
        ):
            s += 0.35
    else:
        s -= 1.6

    if sig.get("willing_to_relocate"):
        s += 0.4
    else:
        if country != "india":
            s -= 0.8

    mode = sig.get("preferred_work_mode")
    if mode in ("hybrid", "flexible"):
        s += 0.25
    elif mode == "onsite":
        s += 0.15
    elif mode == "remote":
        s -= 0.08

    return s


def career_strength(c: Dict) -> float:
    careers = c.get("career_history", [])
    s = 0.0
    txt = norm(
        " ".join(
            (x.get("title", "") + " " + x.get("description", "") + " " + x.get("company", "") + " " + x.get("industry", ""))
            for x in careers
        )
    )
    prod = ["production", "deployed", "implemented", "built", "designed", "owned", "shipped", "real-time", "scale", "scaled", "pipeline", "search", "retrieval", "ranking", "vector", "embedding", "mlops", "experiment"]
    for p in prod:
        if p in txt:
            s += 0.08
    return min(s, 1.2)


def consistency(c: Dict) -> float:
    p = c["profile"]
    y = p.get("years_of_experience", 0) or 0
    career_months = sum(x.get("duration_months", 0) or 0 for x in c.get("career_history", []))
    exp_months = float(y) * 12.0
    s = 0.0

    if exp_months > 0:
        ratio = career_months / exp_months
        if ratio < 0.5 or ratio > 2.0:
            s -= 0.6
        elif ratio < 0.75 or ratio > 1.5:
            s -= 0.2

    tt = norm(p.get("current_title", ""))
    if any(k in tt for k in ["marketing", "support", "accountant", "hr manager", "operations manager"]):
        s -= 0.9

    stext = norm(" ".join(s.get("name", "") for s in c.get("skills", [])) + " " + p.get("summary", ""))
    rel = sum(1 for ph in ["embedding", "retrieval", "ranking", "python", "faiss", "milvus", "qdrant", "weaviate", "pinecone", "elasticsearch", "opensearch", "ndcg", "mrr", "map", "sentence transformer", "llm", "fine-tuning"] if ph in stext)
    if rel == 0:
        s -= 0.6

    if len(c.get("skills", [])) > 25:
        s -= 0.2

    return s


def score_candidate(c: Dict) -> Tuple[float, List[str]]:
    p = c["profile"]
    sig = c.get("redrob_signals", {})
    rel, matched = text_relevance(c)
    s = 0.0
    s += 3.3 * piece_exp_score(p.get("years_of_experience"))
    s += title_score(p.get("current_title", ""))
    s += location_fit(p, sig)
    s += work_fit(sig)
    s += exact_skill_score(c)
    s += rel
    s += career_strength(c)
    s += consistency(c)
    return s, matched


def reason_for_candidate(c: Dict, matched: List[str]) -> str:
    p = c["profile"]
    sig = c.get("redrob_signals", {})
    years = p.get("years_of_experience", 0)
    title = p.get("current_title", "candidate")
    skills = [s.get("name", "") for s in c.get("skills", [])]

    # pick a few highly relevant skills to mention
    preferred = [
        "Sentence Transformers", "Embeddings", "Learning to Rank", "FAISS", "Milvus",
        "Weaviate", "Qdrant", "Pinecone", "Elasticsearch", "OpenSearch",
        "Information Retrieval", "BM25", "Recommendation Systems", "Python",
        "MLOps", "LLM", "Fine-tuning LLMs", "LoRA", "QLoRA", "PEFT"
    ]
    skill_hits = [s for s in preferred if s in skills]
    if not skill_hits:
        # fallback to first few skills
        skill_hits = skills[:3]

    skill_str = ", ".join(skill_hits[:4])
    parts = []
    parts.append(f"{years:.1f} years as {title.lower()} with {skill_str} aligns well with the JD's retrieval/ranking stack.")
    if sig.get("open_to_work_flag"):
        parts.append("The profile is open to work and recently active, which is a strong hiring signal.")
    else:
        parts.append("The profile is not marked open to work, so availability is the main concern.")
    if sig.get("willing_to_relocate"):
        parts.append("Relocation flexibility also fits the Pune/Noida requirement.")
    else:
        country = p.get("country", "")
        if country != "India":
            parts.append(f"Location is {p.get('location', 'unknown')}, so onsite fit would need relocation.")
        else:
            parts.append("The profile does not explicitly signal relocation, so onsite fit is a mild concern.")
    return " ".join(parts[:2]) if len(parts) > 2 else " ".join(parts)


def rank_candidates(candidates_path: Path, out_path: Path, top_n: int = 100) -> None:
    heap: List[Tuple[float, str, Dict, List[str]]] = []

    with open_candidates(candidates_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            score, matched = score_candidate(c)
            item = (score, c["candidate_id"], c, matched)

            # Keep a slightly larger working set so tie-breaking and final sort are stable.
            if len(heap) < 300:
                heapq.heappush(heap, item)
            else:
                # min-heap keyed by score, then reverse candidate_id via lexicographic compare
                if item[0] > heap[0][0] or (item[0] == heap[0][0] and item[1] < heap[0][1]):
                    heapq.heapreplace(heap, item)

    ranked = sorted(heap, key=lambda x: (-x[0], x[1]))[:top_n]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, c, matched) in enumerate(ranked, start=1):
            reason = reason_for_candidate(c, matched)
            writer.writerow([cid, rank, f"{score:.6f}", reason])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    args = parser.parse_args()

    rank_candidates(Path(args.candidates), Path(args.out), top_n=100)


if __name__ == "__main__":
    main()
