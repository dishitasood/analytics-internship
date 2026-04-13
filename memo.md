Voxel: Non-Safety Opportunities & Extraction Quality — Analysis Memo
Prepared by: Dishita Sood
Date: April 2026
Dataset: safety-nonsafety, multiple JSON files


1. Data Quality Assessment
Before trusting any pattern in the labels, I looked at how reliable the extraction itself is.
The extraction output has three consistent failure modes:
    1. Label duplication across phrasings: The same concept appears multiple times under slightly different names: "inventory tracking," "track inventory levels," "real-time inventory visibility." These are all the same use case. 
    2. Safety/non-safety bleed: Several labels appear in both cases across calls, signals the extraction prompt isn't drawing a crisp boundary.
    3. Evidence-label mismatch: Some labels are more specific than the evidence supports.


Confidence rubric I applied:
HIGH confidence: Label + evidence snippet directly match, not ambiguous between safety and non-safety.
MEDIUM confidence: Evidence is indirect or partially supports the label
LOW confidence: Label is generic, evidence is vague

Roughly, I'd estimate 40% of extracted labels are high-confidence, 35% medium, 25% low or noise. 

2. Top 3 Non-Safety Opportunities
Opportunity 1: 
Multiple calls contain customers asking variations of: "How do we know if a line is moving at the rate it should be?" and "We can see when something goes wrong, but not when something is just slow." This is a distinct ask from safety, they want the cameras to detect underperformance. 
It came up across calls with different customer types, and the language was specific. I's a question of what's being detected and surfaced.

Opportunity 2: 
Several calls touched on a need that's adjacent to safety. Proving that procedures were followed. Customers described audit requirements, insurance documentation, and labor compliance. This is a workflow need. This is about what happens after an incident or audit.

Opportunity 3: 
A smaller but consistent signal: customers asking about tracking who's on the floor like contractors, vendors, and visitors. One Representative: "We have contractors coming through all week and we have no visibility into whether they're following our protocols."

3. One Practical Improvement to the Extraction Workflow
Add a mandatory evidence-grounding check before a label is accepted. 
An addition like either a second LLM pass that scores label/evidence alignment on a 1–5 scale, or a human review queue for any label with a confidence score below a threshold would catch the majority of the evidence-label mismatch failures.

The specific implementation I'd suggest: after extraction, pass each (label, evidence) pair to a short scoring prompt: "Does this evidence directly support this use case label? Score 1 (no), 2 (loosely), 3 (maybe), 4(most likely), 5(YES!).

Flag all 1s and 2s for human review. As volume grows, the reviewer focuses only on flagged items, not the full output.

