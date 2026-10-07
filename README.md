# Automated Customer Reviews

This project turns customer reviews into a three-step workflow: sentiment analysis, product-category discovery, and category-level review articles.

## Models

### 1. Sentiment analysis

[Open the sentiment analysis notebook](models/sentiment_analysis_model.ipynb). It fine-tunes **DistilBERT** to classify reviews as **negative**, **neutral**, or **positive** (ratings 1–2, 3, and 4–5). DistilBERT understands context in review text and is compact enough for practical inference. A TF-IDF/logistic-regression baseline is also included for comparison.

**Deployment:** [Try the Streamlit app](https://nlp-automated-customer-reviews-7jbb8pzrk6libeae7frbyo.streamlit.app/)

### 2. Product category clustering

[Open the weighted category clustering notebook](models/category_clustering_model_weighted.ipynb). It encodes category paths, product names, and review text with **Sentence Transformers** (`all-MiniLM-L6-v2`), then applies **K-means** to group products into five categories. Semantic embeddings capture meaning beyond exact word matches, while weighted inputs let product and review evidence contribute to the clusters. The labeled reviews are saved to `data/reviews_with_meta_categories.csv`.

### 3. Category review articles

[Open the Qwen article notebook](models/category_summary_qwen_gpu.ipynb). It uses **Qwen3-4B-Instruct** to draft an article for each category from sampled review evidence and computed product-rating summaries. An instruction-tuned generative model suits the article-writing task; the cited evidence helps readers verify claims. The output is a draft and should be checked against its cited reviews.

## Data

- `data/1429_1.csv` — the source dataset, with about 34,660 customer-review records and 21 columns. It includes product details (such as product name, brand, and category) and review information (such as title, text, rating, recommendation, and date).
- `data/reviews_with_meta_categories.csv` — the same reviews with the five category labels from Model 2. Model 3 uses this file.

The dataset includes products with inconsistent or missing metadata. Category membership and product identity may therefore be uncertain; check the review evidence before treating an article as buying advice.

## Experiments

We also created alternative versions of Models 2 and 3 and compared their results. We did not retain those versions in the main workflow because the selected weighted clustering and Qwen article approaches performed better in our comparisons.

## Run the demo locally

Install the app dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
