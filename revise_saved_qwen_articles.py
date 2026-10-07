"""Render evidence-checked editorial revisions of saved Qwen drafts, without inference.

Original model drafts and prompt-comparison artifacts are retained in original_drafts/.
The JSON sidecars continue to describe the original inference run. Editorial provenance
is recorded separately; revised prose is never labeled as raw model output.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/qwen_category_articles'


def check_citations(text, evidence, profile=None):
    ids = set(re.findall(r'\bE\d+\b', text))
    unknown = ids - evidence.keys()
    if unknown:
        raise ValueError(f'Unknown citations: {sorted(unknown)}')
    if profile is not None:
        wrong = [key for key in ids if evidence[key]['profile_id'] != profile]
        if wrong:
            raise ValueError(f'Citations from another profile in {profile}: {wrong}')


def profile_section(item, summary, evidence, heading):
    profile = int(item['_profile'])
    for text in summary.values():
        check_citations(text, evidence, profile)
    label = summary['inferred_name']
    identity = [f"*Review-derived name.* {summary['name_basis']}", '']
    return [f'### {heading}: {label}', ''] + identity + [
            f"**{item['mean_rating']:.2f}/5** from {item['rated_reviews']} rated reviews; "
            f"1–2-star share: {item['negative_share']:.1%}.", '',
            '**Strengths.** ' + summary['strengths'], '',
            '**Complaints and limits.** ' + summary['complaints'], '']


def render(payload, summary):
    facts = payload['facts']
    evidence = {e['evidence_id']: e for e in payload['evidence']}
    top = facts['top_eligible_profiles']
    low = facts['lowest_rated_eligible_profile']
    for key in ('overview', 'differences', 'lowest', 'conclusion'):
        check_citations(summary[key], evidence)
    for item in top + low:
        profile = int(item['_profile'])
        for text in summary['profiles'][str(profile)].values():
            check_citations(text, evidence, profile)
    for item in low:
        check_citations(summary['lowest'], evidence, int(item['_profile']))
    lines = [f"# {facts['category']}: what customers say", '',
             '> Editorial revision of a saved Qwen draft, checked against the supplied review excerpts. '
             'Statistics come from the saved computed facts; product identities remain unverified.', '',
             f"Based on {facts['source_review_rows']:,} source rows across {facts['metadata_profiles']} metadata profiles. "
             f"Rankings use mean stars among profiles with at least {facts['minimum_rated_reviews']} rated reviews. "
             'Reviews are historical, and the selected excerpts do not establish complaint frequency.', '',
             summary['overview'], '', '## Highest-rated eligible profiles', '']
    if len(top) < 3:
        lines += [f'Only {len(top)} eligible profile(s) are available; a top-three list cannot be supported.', '']
    for index, item in enumerate(top, 1):
        lines += profile_section(item, summary['profiles'][str(item['_profile'])], evidence, f'{index}')
    lines += ['## Key differences', '', summary['differences'], '', '## Lowest-rated eligible profile', '']
    for item in low:
        if item['_profile'] in {p['_profile'] for p in top}:
            lines += [f"Profile {item['_profile']} is the lowest-rated eligible profile; its statistics and review details appear above.", '']
        else:
            lines += profile_section(item, summary['profiles'][str(item['_profile'])], evidence, 'Lowest')
    if not low:
        lines += ['No profile qualifies for this ranking.', '']
    lines += [summary['lowest'], '', '## Conclusion', '', summary['conclusion'], '', '---', '',
              'Evidence IDs refer to the accompanying JSON. Its generation measurements and audits describe the '
              'original Qwen output, preserved in `original_drafts/`. This revision was edited without rerunning Qwen; '
              'editorial provenance is in `editorial_revision_manifest.json`.', '']
    article = '\n'.join(lines)
    # Internal group IDs stay in the evidence and manifest, not consumer prose.
    for profile, details in summary['profiles'].items():
        if 'inferred_name' in details:
            article = re.sub(rf'\bprofile {profile}\b',
                             details['inferred_name'], article, flags=re.I)
    if re.search(r'\bprofile\s+\d+\b', article, re.I):
        raise ValueError('Article still contains an unnamed numeric profile reference.')
    return article


def save_revision(path, article):
    original_dir = path.parent / 'original_drafts'
    original_dir.mkdir(exist_ok=True)
    original = original_dir / path.name
    if not original.exists():
        original.write_bytes(path.read_bytes())
    path.write_text(article, encoding='utf-8')


def main():
    summaries = json.loads((OUTPUT / 'editorial_summaries.json').read_text(encoding='utf-8'))
    manifest = []
    # Validate every article before writing any revisions.
    rendered = []
    for slug, summary in summaries.items():
        payload = json.loads((OUTPUT / f'{slug}.json').read_text(encoding='utf-8'))
        rendered.append((slug, render(payload, summary)))
    for slug, article in rendered:
        save_revision(OUTPUT / f'{slug}.md', article)
        manifest.append({'article': f'{slug}.md', 'original': f'original_drafts/{slug}.md',
                         'evidence': f'{slug}.json', 'method': 'Evidence-checked editorial revision; no new model inference',
                         'review_derived_names': {key: details['inferred_name'] for key, details in summaries[slug]['profiles'].items()},
                         'checks': ['citation existence', 'profile citation attribution', 'statistics rendered from saved facts'],
                         'limits': 'Checks are structural; semantic support was reviewed editorially, not scored independently.'})
    (OUTPUT / 'editorial_revision_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    comparison = ROOT / 'outputs/qwen_prompt_comparison'
    slug = 'cases-and-protective-covers'
    comparison_payload = json.loads((comparison / f'{slug}.json').read_text(encoding='utf-8'))
    save_revision(comparison / f'{slug}.md', render(comparison_payload, summaries[slug]))
    (comparison / 'editorial_revision_manifest.json').write_text(json.dumps({
        'article': f'{slug}.md', 'original': f'original_drafts/{slug}.md',
        'method': 'Evidence-checked editorial revision; no new model inference',
        'comparison_note': 'Use original_drafts/ for the unedited cited-prompt comparison against basic and grounded.'
    }, indent=2), encoding='utf-8')
    print(f'Revised {len(rendered)} category articles and the cited covers article; originals preserved.')


if __name__ == '__main__':
    main()
