# Qwen GPU experiment: results and limitations

## Editorial revision of saved articles

All ranked groups now have review-derived names in the revised articles, including Echo, chargers, and tablets. Numeric profile references have been removed from the reader-facing prose; the editorial manifest maps internal IDs to the displayed names. Generic descriptive labels are used when the reviews do not establish an exact model. These labels describe the reviewed item or product family, not verified catalog identities. The Fire TV label covers mixed models because its reviews mention a box, FireTV-2, and a Stick. The HDX charger label describes advertised compatibility rather than asserting a brand or successful compatibility. The PowerFast name uses the full saved review E28633; its original prompt excerpt alone does not include all the identifying context. Source metadata, group membership, and ratings were not changed.

The current five category articles and the cited covers article in `../qwen_prompt_comparison/` were edited against the saved review excerpts without rerunning Qwen. They include profile-specific strengths and complaints, supported differences, and reasons to investigate the lowest-rated eligible profile. Ratings and counts are rendered directly from the saved computed facts. The covers rating and complaint attribution errors described below have been corrected in these revisions. Unsupported quotes were replaced with evidence-based paraphrases, and a positive-text one-star tablet review is explicitly identified as unsuitable complaint evidence.

Original model prose is retained in each directory's `original_drafts/`. The measurements, original-generation audits, and observations below concern those original drafts. Editorial provenance is recorded in `editorial_revision_manifest.json`; the JSON inference sidecars were left intact. All revised citation IDs and profile-specific attribution passed structural checks. This is not a new model benchmark or an independent factuality score. Product identities remain unverified.

To reproduce the editorial revisions from the existing local evidence sidecars, run `python revise_saved_qwen_articles.py`. The summaries used by that renderer are stored in `editorial_summaries.json`. The Qwen cited prompt was also updated for future runs to request profile-level sections and leave numerical ratings to the computed table; that new prompt has not yet been evaluated on GPU.

Generated locally on NVIDIA GeForce RTX 5060 Laptop GPU (8 GB), using PyTorch 2.11.0+cu128 and 4-bit NF4 Qwen3-4B-Instruct. The original weights are pinned to revision `cdbee75f17c01a7cc42f958dc650907174af0554`. GPU computation and 4-bit kernels were tested before generation.

## Generation measurements

| Category | Input tokens | Output tokens | Seconds | Peak GPU allocation (GiB) |
|---|---:|---:|---:|---:|
| Cases and protective covers | 3496 | 444 | 45.33 | 6.626 |
| Chargers, cables and power accessories | 2674 | 446 | 43.49 | 5.058 |
| Echo speakers and smart-home devices | 3578 | 432 | 84.73 | 6.802 |
| Fire TV and streaming devices | 3266 | 492 | 46.73 | 6.151 |
| Tablets and e-readers | 3867 | 437 | 93.0 | 7.455 |

Peak allocation is reported by PyTorch, not total system GPU use. The largest prompt approached the GPU memory limit. All five outputs finished before the token limit, and all cited evidence IDs exist. Citation existence does not establish claim support.

## Observed quality

The Qwen articles are shorter and more organized than the BART drafts inspected here, with source references and reviewer attribution. This is qualitative inspection, not a controlled model benchmark; the evidence samples differ.

- Covers: Qwen gives profile 80 a mean rating of 3.0; the computed value is 3.92. It also associates complaint E28660 with profile 86, although that review belongs to profile 80. Use the computed table for statistics.
- Echo: the quoted phrase ?subscription trap? is absent from the evidence excerpts supplied to the model. Other articles also contain quoted phrases not found verbatim in those excerpts. These flags are recorded in each evidence JSON.
- Missing and contradictory metadata names still prevent verified named-product recommendations. Category summaries and cautious profile comparisons are supported more reliably.

## Three-prompt comparison

Basic, grounded and cited prompts were run on identical cover-category facts and review excerpts. Basic and grounded variants write more continuous prose but collapse several distinct profiles into a single case or cover. The cited variant exposes its sources and is more cautious about ranking, but still makes factual mistakes. The cited prompt is retained for traceability, not because factual superiority has been demonstrated. See `../qwen_prompt_comparison/index.md`.

## Checks and limits

Source-row positions, full review text, category/profile attribution in the evidence files, model pin, 4-bit runtime, output completion and rating tables were checked. Quote/rating audits are limited heuristics, not comprehensive factuality checks. No fine-tuning or independent accuracy score is claimed. Verify product identities and manually check each claim before publishing buying advice.
