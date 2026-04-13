What I did
I started by reading through the JSON files manually before touching any code. That pass gave me a working theory of what the data quality issues actually were before I tried to write anything systematic.
From there I wrote a Python script that:

Loads all the JSON files from safety-nonsafety/
Flattens and normalizes use case labels
Uses sentence-transformer embeddings + cosine similarity to cluster near-duplicates
Exports a cleaned CSV for inspection

I deliberately kept the code simple, no LangChain, no crazy pipeline. The goal was something I could hand to someone and they'd understand it in five minutes.

Key decisions
Jupyter vs. plain Python: I went with a standalone .py script rather than a notebook. Notebooks are great for exploration, but a script is easier to read linearly, version-control cleanly, and review quickly. If Voxel uses notebooks internally, I'd swap to that format immediately.

Embeddings over pure fuzzy matching: Simple edit-distance would catch "track inventory" vs "inventory tracking" but would miss semantic near-duplicates. Sentence transformers handle both. The tradeoff is a heavier dependency and slightly slower runtime on a large corpus.

What I included in the memo vs. the code: The code shows methodology. The memo is where I made judgment calls: which clusters are signal vs. noise, what the extraction pipeline is probably getting wrong, and what to do about it.


Assumptions

I treated each JSON file as one call.
I assumed the extraction pipeline was LLM-based.

## WHERE I USED CLAUDE

**Initial scaffolding + structure**

Used Claude to sketch out the overall script structure, what functions were needed, how to break up loading vs normalization vs clustering vs output. The design decisions (greedy vs k-means, embeddings vs fuzzy) were mine.

**Updating extract_use_cases after seeing real JSON**

The first version assumed a flat evidence string. Once I shared the actual JSON structure, Claude updated the parsing logic. Also added the speaker parsing and evidence_source logic at that point.

**Analytical observations on specific files**

Pasted in the Signal Grid and West Drive files and talked through what was interesting. The deployment blocker insight came out of that conversation, I'd noticed the quote but Claude helped figure out why it was a failure mode worth flagging.

**Spotting bugs in the version I'd edited locally**

After making manual edits I pasted the full file back in. Claude caught the minor errors, the formal_evidence → format_evidence mismatch, and confirmed which bugs still needed my attention. Faster than re-reading the whole file myself.