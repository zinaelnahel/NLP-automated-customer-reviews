"""Task 3: local BART review summaries with auditable product statistics."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = 'sshleifer/distilbart-cnn-12-6'
MODEL_REVISION = 'a4f8f3ea906ed274767e9906dbaede7531d660ff'
GENERATION_SETTINGS = {'max_new_tokens': 100, 'min_length': 0, 'num_beams': 2,
                       'do_sample': False, 'no_repeat_ngram_size': 3}
PROFILE_COLUMNS = ['id', 'name', 'brand', 'categories']


def prepare_data(path):
    df = pd.read_csv(path, low_memory=False)
    required = PROFILE_COLUMNS + ['meta_category', 'reviews.text', 'reviews.title', 'reviews.rating']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df['meta_category'].isna().any():
        raise ValueError('Every review needs a category from Task 2.')
    df['_row'] = range(len(df))
    df['_text'] = df[['reviews.title', 'reviews.text']].fillna('').agg(' '.join, axis=1).str.replace(r'\s+', ' ', regex=True).str.strip()
    df['_rating'] = pd.to_numeric(df['reviews.rating'], errors='coerce')
    df.loc[~df['_rating'].between(1, 5), '_rating'] = float('nan')
    df['_profile'] = df.groupby(PROFILE_COLUMNS, dropna=False, sort=False).ngroup()
    if df.groupby('_profile')['meta_category'].nunique().gt(1).any():
        raise ValueError('A metadata profile belongs to multiple categories.')
    return df


def product_stats(df):
    stats = df.groupby('_profile').agg(
        name=('name', 'first'), reviews=('_row', 'size'),
        rated_reviews=('_rating', 'count'), mean_rating=('_rating', 'mean'),
        negative_reviews=('_rating', lambda s: int(s.le(2).sum())),
    )
    stats['negative_share'] = stats['negative_reviews'] / stats['rated_reviews'].replace(0, float('nan'))
    stats['name'] = stats['name'].fillna('[missing product name]').map(lambda s: re.sub(r'\s+', ' ', str(s)).strip(' ,'))
    return stats


def select_evidence(df, limit=8):
    """Round-robin profiles; prefer helpful reviews within each profile."""
    rows = df[df['_text'].ne('')].drop_duplicates('_text').copy()
    rows['_helpful'] = pd.to_numeric(rows.get('reviews.numHelpful', pd.Series(0, index=rows.index)), errors='coerce').fillna(0)
    rows = rows.sort_values(['_helpful', '_row'], ascending=[False, True])
    rows['_turn'] = rows.groupby('_profile').cumcount()
    return rows.sort_values(['_turn', '_helpful', '_row'], ascending=[True, False, True]).head(limit)


def clean_summary(text):
    return re.sub(r'\s+([.,!?;:])', r'\1', re.sub(r'\s+', ' ', text)).strip()


class BartSummarizer:
    def __init__(self, model_id=MODEL_ID, variant='plain'):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        revision = MODEL_REVISION if model_id == MODEL_ID else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision).eval()
        self.model.generation_config.max_length = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.variant = variant
        self.revision = getattr(self.model.config, '_commit_hash', None)
        self.cache_dir = ROOT / 'outputs' / 'bart_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def summarize(self, texts, topic):
        if not texts:
            return 'No non-empty reviews available for this section.'
        # BART is a summarizer, not an instruction-tuned chatbot. Compare source
        # framing, rather than assuming it follows elaborate instructions.
        source = ' '.join(texts)
        if self.variant == 'framed':
            source = f'Customer reviews discussing {topic}. ' + source
        key = hashlib.sha256(json.dumps([self.model.config._name_or_path, self.revision, GENERATION_SETTINGS, self.variant, source]).encode()).hexdigest()
        cache = self.cache_dir / f'{key}.txt'
        if cache.exists():
            return clean_summary(cache.read_text(encoding='utf-8'))
        tokens = self.tokenizer.encode(source, add_special_tokens=False, verbose=False)
        # Process all selected evidence, with no silent truncation at 1024 tokens.
        budget = min(900, self.model.config.max_position_embeddings - 2)
        summaries = []
        for start in range(0, len(tokens), budget):
            chunk = [self.tokenizer.bos_token_id] + tokens[start:start + budget] + [self.tokenizer.eos_token_id]
            input_ids = self.torch.tensor([chunk], device=self.device)
            batch = {'input_ids': input_ids, 'attention_mask': self.torch.ones_like(input_ids)}
            with self.torch.inference_mode():
                result = self.model.generate(**batch, **GENERATION_SETTINGS,
                                             forced_bos_token_id=self.tokenizer.bos_token_id)
            summaries.append(self.tokenizer.decode(result[0], skip_special_tokens=True))
        # Joining chunk summaries preserves minority complaints instead of a
        # second reduction that might discard them.
        summary = clean_summary(' '.join(summaries))
        cache.write_text(summary, encoding='utf-8')
        return summary


def generate_articles(df, summarizer, output_dir, min_reviews=10, category=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label, rows in df.groupby('meta_category', sort=True):
        if category and label != category:
            continue
        print(f'Generating: {label}', flush=True)
        stats = product_stats(rows)
        eligible = stats[stats['rated_reviews'].ge(min_reviews)]
        top = eligible.sort_values(['mean_rating', 'rated_reviews'], ascending=False).head(3)
        worst = eligible.sort_values(['mean_rating', 'rated_reviews'], ascending=[True, False]).head(1)
        lines = [f'# {label}: what customers say', '',
                 '> BART experiment draft. The prose reports sampled customer opinions; product identities and technical claims have not been independently verified.', '',
                 f'Based on {len(rows):,} source review rows across {len(stats)} metadata profiles. '
                 'Names and product metadata may be inconsistent; these are profile-level comparisons, not verified model identities.', '']
        evidence = []

        def section(title, subset, topic):
            selected = select_evidence(subset)
            lines.extend([f'### {title}', '', summarizer.summarize(selected['_text'].tolist(), topic), ''])
            evidence.extend({'section': title, 'source_row': int(r['_row']), 'profile_id': int(r['_profile']),
                             'rating': None if pd.isna(r['_rating']) else float(r['_rating']), 'text': r['_text']}
                            for _, r in selected.iterrows())

        section('Category overview: positive experiences', rows[rows['_rating'].ge(4)], 'positive experiences')
        section('Category overview: complaints', rows[rows['_rating'].le(2)], 'complaints')
        lines.extend(['## Top products and differences', '',
                      f'Ranked by mean star rating, with at least {min_reviews} rated reviews per profile. '
                      'This is a descriptive ranking of this dataset, not a purchase recommendation.', ''])
        if top.empty:
            lines.extend(['Insufficient rated reviews to rank products at this threshold.', ''])
        for profile, item in top.iterrows():
            lines.extend([f"### {item['name']} (profile {profile})", '',
                          f"Mean rating: {item['mean_rating']:.2f}/5 from {int(item['rated_reviews'])} rated reviews; "
                          f"1–2-star share: {item['negative_share']:.1%}.", ''])
            product = rows[rows['_profile'].eq(profile)]
            section('What reviewers like', product[product['_rating'].ge(4)], 'product strengths')
            section('Reported complaints', product[product['_rating'].le(2)], 'product complaints')
        lines.extend(['## Lowest-rated eligible profile', ''])
        if worst.empty:
            lines.extend(['Insufficient evidence to identify a lowest-rated product.', ''])
        else:
            profile = worst.index[0]
            item = worst.iloc[0]
            lines.extend([f"{item['name']} (profile {profile}) has the lowest mean rating among eligible profiles: "
                          f"{item['mean_rating']:.2f}/5 from {int(item['rated_reviews'])} ratings. "
                          'A relative last place does not establish that a product should be avoided.', ''])
            section('Reasons to investigate before buying', rows[rows['_profile'].eq(profile) & rows['_rating'].le(2)], 'negative experiences')
        lines.extend(['## Reading this article', '',
                      'Prose is generated by BART from a bounded, deduplicated review sample; numbers are computed from all source rows. '
                      'Duplicate source reviews can affect rankings. Missing ratings do not enter ranking or sentiment-specific samples. '
                      'Generated claims require manual checking against the accompanying evidence JSON. '
                      'No fine-tuning or independent quality validation has been performed.', ''])
        slug = re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')
        (output_dir / f'{slug}.md').write_text('\n'.join(lines), encoding='utf-8')
        payload = {'category': label, 'model': MODEL_ID, 'model_revision': summarizer.revision,
                   'variant': summarizer.variant, 'min_rated_reviews': min_reviews,
                   'generation_settings': GENERATION_SETTINGS,
                   'source_row_convention': 'Zero-based data-row position in input CSV, excluding the header.',
                   'source_reviews': len(rows), 'evidence': evidence,
                   'product_statistics': json.loads(stats.reset_index().to_json(orient='records'))}
        (output_dir / f'{slug}.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        manifest.append({'category': label, 'article': f'{slug}.md', 'evidence': f'{slug}.json'})
    if not manifest:
        raise ValueError('No matching categories found.')
    (output_dir / 'index.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    index_lines = ['# BART category article drafts', '',
                   'Generated locally from Task 2 categories. Review evidence JSON files are generated locally and excluded from Git.', '']
    index_lines.extend(f"- [{item['category']}]({item['article']}) · [Evidence]({item['evidence']})" for item in manifest)
    (output_dir / 'index.md').write_text('\n'.join(index_lines) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/reviews_with_meta_categories.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/bart_category_articles')
    parser.add_argument('--category', help='Generate only this exact category name for a quick experiment.')
    parser.add_argument('--variant', choices=['plain', 'framed'], default='plain')
    parser.add_argument('--min-reviews', type=int, default=10)
    args = parser.parse_args()
    if args.min_reviews < 1:
        parser.error('--min-reviews must be positive')
    df = prepare_data(args.data)
    summarizer = BartSummarizer(variant=args.variant)
    generate_articles(df, summarizer, args.output, args.min_reviews, args.category)


if __name__ == '__main__':
    main()
