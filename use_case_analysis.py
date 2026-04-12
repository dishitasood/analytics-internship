"""
use_case_analysis.py
Voxel Research & Analytics Internship Take-Home
 
Loads JSON call files, normalizes use case labels, clusters near-duplicates
using sentence embeddings, and exports a cluster summary CSV.
 
Dependencies: sentence-transformers, scikit-learn, pandas
Install: pip install sentence-transformers scikit-learn pandas

Example JSON structure for understanding:
{
  "meeting_title": "...",
  "start_time": "...",
  "extraction": {
    "safety_use_cases": [
      {
        "label": "...",
        "description": "...",
        "evidence": [
          {"quote": "...", "speaker": "Name <email>", "timestamp": "[HH:MM:SS]"},
          ...
        ]
      }
    ],
    "nonsafety_use_cases": [ ... ]   ← may be empty list []
  }
}

"""
import json
import os
import re
import csv
import pathlib as Path
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configuration
DATA_DIR = Path("../safety-nonsafety")
OUTPUT_CLUSTERS = Path("clusters.csv")
SIMILARITY_THRESHOLD = 0.82

EMBED_MODEL = "all-MiniLM-L6-v2"


# Use cases that are almost certainly extraction noise — too generic to be
# actionable product signal. Exact match against normalized label.
JUNK_LABELS = {
    "data reporting",
    "administrative oversight",
    "workflow management",
    "general monitoring",
    "operational visibility",
    "data collection",
    "reporting",
    "monitoring",
    "management",
    "oversight",
}

# Terms with only these words will be suggested as noise
VAGUE_SINGLE_TOKENS = {"monitoring", "tracking", "management", "reporting", "visibility"}


# Phrases in evidence quotes that suggest a deployment blocker —
# the use case may be aspirational or constrained rather than active.
BLOCKER_PHRASES = [
    "union wouldn't allow",
    "union would not allow",
    "can't do that",
    "cannot do that",
    "not currently supported",
    "future scope",
    "would have to be trained",
    "outdoor only",
    "requiring indoor training",
    "privacy concern",
    "legal concern",
]

# Text normalization
def normalize_label(abel: str) -> str:
    """
    Lowercase, strip punctuation, remove filler phrases, standardize word order.
    We intentionally do NOT stem (e.g., 'tracks' → 'track') because stemming
    makes labels harder to read in output and sentence embeddings handle
    morphological variation already.
    """
    label = label.lower().strip()
    for filler in ["real-time", "real time", "automated", "automatic"]:
        label = label.replace(filler, "")
    label = re.sub(r"[a-z0-9 ]", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def is_junk(label:str) -> bool:
    """Flag labels too generic to be useful signal."""

    if label in JUNK_LABELS:
        return True
    
    tokens = set(label.split())
    if len(tokens) <= 1 and tokens <= VAGUE_SINGLE_TOKENS:
        return True
    return False


def extract_speaker_name(speaker_str: str) ->str:
    """Parse 'Name Last_name <email>' -> 'Name Last_name'.
    Falls back to the raw string if parsing fails.
    """

    return speaker_str.split("<")[0].strip() if "<"in speaker_str else speaker_str.strip()


def is_voxel_speaker(speaker_str: str) -> bool:
    """
    If the email domain contains 'voxel', the speaker is internal
    Useful for distinguishing customer signal from Voxel rep pitching
    """

    return "voxelai.com" in speaker_str.lower() or "voxel.com" in speaker_str.lower()

def formal_evidence(evidence_list: list) -> str:
    """
    Flatten a list of evidence objects into a readable string:
    'Speaker Name: quote text | Speaker Name: quote text'
 
    Handles the confirmed structure: [{"quote": ..., "speaker": ..., "timestamp": ...}]
    Also handles legacy plain-string evidence just in case.
    """

    if not evidence_list:
        return ""
    
    parts = []
    for e in evidence_list:
        if isinstance(e, dict):
            speaker = extract_speaker_name(e.get("speaker", "Unknown"))
            quote = e.get("quote", "").strip()
            if quote:
                parts.append(f"{speaker}: {quote}")

        elif isinstance(e, str) and e.strip():
            parts.appned(e.strip())

    return " | ".join(parts)

def has_deployment_blocker(evidence_list: list) -> bool:
    """
    Scan evidence quotes for language that suggests the use case has a
    constraint or blocker (union rules, technical limitations, future scope).
    These labels shouldn't be treated as active use cases.
    """

    for e in evidence_list:
        quote = ""
        if isinstance(e, dict):
            quote = e.get("quote", "").lower()
        elif isinstance(e, str):
            quote = e.lower()
        if any(phrase in quote for phrase in BLOCKER_PHRASES):
            return True
        
    return False

def evidence_source(evidence_list: list) -> str:
    """
    Returns 'customer', 'voxel', or 'mixed' depending on who surfaces
    the use case in the evidence. Matters for signal quality:
    a use case raised by the customer is stronger signal than one pitched
    by the Voxel rep.
    """
    speakers = [
        e.get("speaker", "") for e in evidence_list if isinstance(e, dict)
    ]
    if not speakers:
        return "unknown"
    voxel = sum(1 for s in speakers if is_voxel_speaker(s))
    customer = len(speakers) - voxel
    if voxel > 0 and customer > 0:
        return "mixed"
    return "voxel" if voxel > 0 else "customer"


# ── Data loading ──────────────────────────────────────────────────────────────
 
def load_call(filepath: Path) -> dict:
    """Load a single JSON call file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)
 
 
def extract_use_cases(data: dict, source_file: str) -> list[dict]:
    """
    Extract all use cases from one call's JSON into a flat list of records.
 
    Key decisions:
    - Unwrap 'extraction' key (confirmed present in all real files)
    - Fallback to top-level if 'extraction' key is missing (defensive)
    - Preserve bucket (safety vs nonsafety) as metadata for bleed analysis
    - Capture description separately from evidence — it's richer than the label
      but was LLM-generated, so treat it as supplementary not ground truth
    - Flag deployment blockers at record level so they survive into cluster summary
    - Tag evidence source (customer vs voxel) as a signal quality indicator
    """
    records = []
 
    # Unwrap extraction key — confirmed structure in all three sample files
    extraction = data.get("extraction", data)
 
    meeting_title = data.get("meeting_title", "")
    start_time = data.get("start_time", "")
 
    for bucket in ["safety_use_cases", "nonsafety_use_cases"]:
        items = extraction.get(bucket, [])  # empty list if bucket missing or []
 
        for item in items:
            if isinstance(item, str):
                # Fallback: plain string label, no evidence
                label = item
                description = ""
                evidence_list = []
            elif isinstance(item, dict):
                label = (
                    item.get("label")
                    or item.get("use_case")
                    or item.get("name")
                    or ""
                )
                description = item.get("description", "")
                evidence_list = item.get("evidence", [])
            else:
                continue
 
            if not label:
                continue
 
            norm = normalize_label(label)
            evidence_str = format_evidence(evidence_list)
            blocker = has_deployment_blocker(evidence_list)
            source = evidence_source(evidence_list)
            evidence_count = len(evidence_list)
 
            records.append({
                "source_file": source_file,
                "meeting_title": meeting_title,
                "start_time": start_time,
                "bucket": bucket,
                "raw_label": label,
                "normalized_label": norm,
                "description": description,
                "evidence": evidence_str,
                "evidence_count": evidence_count,       # proxy for extraction confidence
                "evidence_source": source,              # customer / voxel / mixed / unknown
                "has_deployment_blocker": blocker,      # union rules, tech limits, future scope
                "is_junk": is_junk(norm),
            })
 
    return records
 
 
def load_all_calls(data_dir: Path) -> list[dict]:
    """Load every JSON file in data_dir. Returns flat list of use case records."""
    all_records = []
    json_files = sorted(data_dir.glob("*.json"))
 
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
 
    print(f"Loading {len(json_files)} call files...")
    for fp in json_files:
        try:
            data = load_call(fp)
            records = extract_use_cases(data, fp.name)
            all_records.extend(records)
            print(f"  {fp.name}: {len(records)} use cases extracted")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Could not parse {fp.name}: {e}")
 
    return all_records
 
 
def load_all_calls(data_dir: Path) -> list[dict]:
    """Load every JSON file in data_dir, return flat list of use case records."""
    all_records = []
    json_files = sorted(data_dir.glob("*.json"))
 
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
 
    print(f"Loading {len(json_files)} call files...")
    for fp in json_files:
        try:
            data = load_call(fp)
            records = extract_use_cases(data, fp.name)
            all_records.extend(records)
            print(f"  {fp.name}: {len(records)} use cases")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Could not parse {fp.name}: {e}")
 
    return all_records



# ── Clustering ─────────────────────────────────────────────────────────────────

def cluster_use_cases(records: list[dict], threshold: float) -> list[dict]:
    """
    Embed normalized labels and greedily cluster by cosine similarity.
 
    Strategy: greedy single-pass clustering. For each label (sorted by
    frequency descending so common ones become cluster centroids), assign
    it to the first existing cluster where similarity > threshold.
    If no match, start a new cluster.
 
    Why greedy over k-means: we don't know k in advance, and k-means
    centroids drift away from readable labels. Greedy keeps cluster names
    human-readable (they're actual labels from the data).
 
    Tradeoff: greedy is order-sensitive. Sorting by frequency first
    mitigates this — common labels are more likely to be "canonical."
    """

    # Filter junk before clustering
    clean = [r for r in records if not r["is_junk"]]

    if not clean:
        print("WARNING: All labels filtered as junk. Check JUNK_LABELS config.")
        return []
    
    # Count frequency to use as sort key
    label_counts = defaultdict(int)
    for r in clean:
        label_counts[r["normalized_label"]] += 1

    # Get unique labels sorted by frequency
    unique_labels = sorted(set(r["normalized_label"] for r in clean), key=lambda l: label_counts[l], reverse=True)

    print(f"\nEmbedding {len(unique_labels)} unique labels with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(unique_labels, show_progress_bar=True)

    # Greedy clustering
    clusters = {}      # cluster_id → {"centroid_label", "members": [...], "embedding"}

    label_to_cluster = {}  # normalized_label → cluster_id
 
    for i, label in enumerate(unique_labels):
        emb = embeddings[i]
        best_cluster = None
        best_sim = 0.0
 
        for cid, cluster in clusters.items():
            sim = cosine_similarity([emb], [cluster["embedding"]])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_cluster = cid
 
        if best_cluster is not None and best_sim >= threshold:
            clusters[best_cluster]["members"].append(label)
            label_to_cluster[label] = best_cluster
        else:
            new_id = len(clusters)
            clusters[new_id] = {
                "centroid_label": label,
                "members": [label],
                "embedding": emb,
                "count": 0,
            }
            label_to_cluster[label] = new_id
 
    # Attach cluster info back to records
    for r in clean:
        cid = label_to_cluster.get(r["normalized_label"])
        if cid is not None:
            r["cluster_id"] = cid
            r["cluster_name"] = clusters[cid]["centroid_label"]
            clusters[cid]["count"] += 1
        else:
            r["cluster_id"] = -1
            r["cluster_name"] = "unclustered"
 
    return clean, clusters

# ── Output ─────────────────────────────────────────────────────────────────────
def confidence_score(records: list[dict], bleed: bool) -> str:
    """
    Simple 3-tier confidence rubric per cluster:
 
    HIGH:   appears in 3+ calls, customer-raised, no blocker, no bleed
    MEDIUM: appears in 2 calls OR has bleed OR mixed evidence source
    LOW:    single call, voxel-only source, or has deployment blocker
 
    This is a heuristic — use it to prioritize human review, not as ground truth.
    """

    distinct_calls = len(set(r["source_file"] for r in records))
    has_blocker = any(r.get("has_deployment_blocker") for r in records)
    sources = set(r.get("evidence_source", "unknown") for r in records)
    customer_raised = "customer" in sources or "mixed" in sources

    if has_blocker:
        return "LOW"
    if distinct_calls >= 3 and customer_raised and not bleed:
        return "HIGH"
    if distinct_calls >= 2 or (customer_raised and not bleed):
        return "MEDIUM"
    return "LOW"

def build_cluster_summary(clean_records: list[dict], clusters: dict) -> pd.DataFrame:
    """
    Build summary DataFrame: one row per cluster.
 
    Columns:
    - cluster_name: most frequent normalized label (human-readable centroid)
    - total_mentions: raw count across all records
    - distinct_calls: how many unique files this appeared in (stronger signal than raw count)
    - safety_nonsafety_bleed: True if cluster spans both buckets (extraction ambiguity flag)
    - has_deployment_blocker: True if any evidence quote mentions constraints
    - evidence_source: customer / voxel / mixed (customer-raised = stronger signal)
    - confidence: HIGH / MEDIUM / LOW rubric
    - member_labels: all deduplicated variant phrasings in this cluster
    - evidence_sample: one representative quote for quick human review
    """

    rows = []
    for cid, cluster in clusters.items():
        members_records = [r for r in clean_records if r.get("cluster_id") == cid]
        if not members_records:
            continue
 
        files = set(r["source_file"] for r in members_records)
        buckets = set(r["bucket"] for r in members_records)
        bleed = len(buckets) > 1
 
        has_blocker = any(r.get("has_deployment_blocker") for r in members_records)
 
        # Aggregate evidence sources across cluster
        sources = set(r.get("evidence_source", "unknown") for r in members_records)
        if "customer" in sources and "voxel" in sources:
            agg_source = "mixed"
        elif "customer" in sources:
            agg_source = "customer"
        elif "mixed" in sources:
            agg_source = "mixed"
        else:
            agg_source = "voxel"
 
        # Pick the single strongest evidence sample: prefer customer-raised,
        # highest evidence_count, no blocker
        best = sorted(
            [r for r in members_records if r.get("evidence")],
            key=lambda r: (
                r.get("evidence_source") == "customer",
                r.get("evidence_count", 0),
                not r.get("has_deployment_blocker", False),
            ),
            reverse=True,
        )
        evidence_sample = best[0]["evidence"][:300] if best else ""
 
        conf = confidence_score(members_records, bleed)
 
        rows.append({
            "cluster_name": cluster["centroid_label"],
            "total_mentions": len(members_records),
            "distinct_calls": len(files),
            "confidence": conf,
            "safety_nonsafety_bleed": bleed,
            "has_deployment_blocker": has_blocker,
            "evidence_source": agg_source,
            "member_labels": " | ".join(sorted(set(cluster["members"]))),
            "evidence_sample": evidence_sample,
        })
 
    df = pd.DataFrame(rows)
    # Sort: distinct calls first, then confidence tier
    conf_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    df["_conf_sort"] = df["confidence"].map(conf_order)
    df = df.sort_values(
        ["distinct_calls", "_conf_sort"], ascending=[False, True]
    ).drop(columns=["_conf_sort"]).reset_index(drop=True)
    return df

def print_top_clusters(df: pd.DataFrame, n: int = 15):
    print(f"\n{'-'*70}")
    print(f"TOP {n} CLUSTERS BY DISTINCT CALL COUNT")
    print(f"{'-'*70}")

    for _, row in df.head(n).iterrows():
        flags = []
        if row["safety_nonsafety_bleed"]:
            flags.append("⚠ BLEED")
        if row["has_deployment_blocker"]:
            flags.append("⚠ BLOCKER")
        
        flag_str = "  " + "  ".join(flags) if flags else ""

        print(
            f"\n[{row['distinct_calls']} calls | {row['total_mentions']} mentions"
            f" | {row['confidence']} confidence | source: {row['evidence_source']}]"
            f"{flag_str}"
        )

        print(f"  {row['cluster_name']}")
        print(f"  Members: {row['member_labels']}")
        if row["evidence_sample"]:
            print(f"  Evidence: {row['evidence_sample'][:150]}...")


# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    # 1. Load all calls
    all_records = load_all_calls(DATA_DIR)
    print(f"\nTotal use case records extracted: {len(all_records)}")
    junk_count = sum(1 for r in all_records if r["is_junk"])
    print(f"Junk filtered: {junk_count}")
    blocker_count = sum(1 for r in all_records if r.get("has_deployment_blocker"))
    print(f"Records with deployment blockers: {blocker_count}")
 
    # 2. Cluster (on non-junk records only)
    clean_records, clusters = cluster_use_cases(all_records, SIMILARITY_THRESHOLD)
    print(f"Unique clusters formed: {len(clusters)}")
 
    # 3. Build summary
    df = build_cluster_summary(clean_records, clusters)
 
    # 4. Save CSV
    df.to_csv(OUTPUT_CLUSTERS, index=False)
    print(f"\nCluster summary saved to: {OUTPUT_CLUSTERS}")
 
    # 5. Print top clusters
    print_top_clusters(df)
 
    # 6. Summary stats
    nonsafety = [r for r in clean_records if r["bucket"] == "nonsafety_use_cases"]
    bleed_clusters = df[df["safety_nonsafety_bleed"] == True]
    blocker_clusters = df[df["has_deployment_blocker"] == True]
    customer_raised = df[df["evidence_source"].isin(["customer", "mixed"])]
 
    print(f"\n{'─'*70}")
    print(f"SUMMARY STATS")
    print(f"{'─'*70}")
    print(f"  Calls processed:                    {len(set(r['source_file'] for r in all_records))}")
    print(f"  Total use case records:             {len(all_records)}")
    print(f"  After junk filter:                  {len(clean_records)}")
    print(f"  Non-safety records:                 {len(nonsafety)}")
    print(f"  Clusters (total):                   {len(df)}")
    print(f"  Clusters w/ safety/nonsafety bleed: {len(bleed_clusters)}")
    print(f"  Clusters w/ deployment blocker:     {len(blocker_clusters)}")
    print(f"  Clusters customer-raised:           {len(customer_raised)}")
    print(f"  HIGH confidence clusters:           {len(df[df['confidence']=='HIGH'])}")
    print(f"  MEDIUM confidence clusters:         {len(df[df['confidence']=='MEDIUM'])}")
    print(f"  LOW confidence clusters:            {len(df[df['confidence']=='LOW'])}")
 
 
if __name__ == "__main__":
    main()
 
