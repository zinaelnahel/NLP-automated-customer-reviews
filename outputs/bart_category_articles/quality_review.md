# BART baseline: observed quality

All five articles were generated locally with the pinned DistilBART checkpoint on CPU. No fine-tuning was performed. Statistics and source attribution were checked programmatically; prose was inspected qualitatively. This is not an independent accuracy measurement.

## Source-framing experiment

`source_framing_comparison.json` contains two outputs from identical selected low-rating cover reviews and identical decoding: raw text versus a descriptive introductory sentence. Both retain the hinge/cracking complaint. Framing adds thin-leather and floppy-cover details, while the raw version retains sleep-mode alignment issues. Neither establishes a clear overall winner, and both repeat a reviewer's prediction about cracking as a categorical claim. Keep the plain baseline for reproducibility, not because it has demonstrated superior quality.

## Observed problems

- Streaming: the generated overview attributes a remark to ?CNN's John Sutter.? The selected evidence mentions CNN, but contains no John Sutter attribution. This is an unsupported addition consistent with the checkpoint's news-summarization domain.
- Covers: the ranked eligible profiles have missing names. Only two profiles satisfy the ten-rating minimum. They cannot be presented as verified named product recommendations.
- Streaming: only one profile satisfies the threshold. It is simultaneously the highest- and lowest-rated eligible profile; that does not justify avoiding it.
- Chargers and Echo: several metadata names describe a different device type from their reviews. The generated text follows review content, so the named-product comparisons remain unreliable.
- Tablets: some low-rating reviews contain positive or generic wording. A star-based complaint sample is not guaranteed to contain an actual complaint. Some output also retains awkward phrases and questionable specifications.
- Positive-review sections can include criticisms because even highly rated reviews discuss tradeoffs. Do not interpret the heading as proof that every sentence expresses praise.

## Decision and next steps

Use these articles as an experimental baseline and review drafts, not as published buying advice. First repair or verify product identities. Then create human-reviewed summaries and compare framing/decoding variants on a held-out set with factuality, product attribution, complaint coverage, and readability criteria. If fine-tuning is needed, split by verified product identity to avoid leakage. A larger model alone does not fix inconsistent metadata.
