# Customer Review Sentiment Analysis

A fine-tuned DistilBERT classifier and public web demo for labeling product reviews as **negative**, **neutral**, or **positive**. The model takes review text as input and returns its predicted class with probabilities for all three classes.

## Model and data

Star ratings are converted into training labels: 1–2 stars are negative, 3 stars are neutral, and 4–5 stars are positive. Review title and body provide the text; the rating is never passed to the model as an input. Training uses only `data/1429_1.csv`, with stratified train, validation, and test splits. Inverse-frequency loss weights account for class imbalance, and validation macro F1 selects the saved checkpoint.

The latest three-epoch run selected epoch 2 by validation macro F1 and achieved **93.65% test accuracy** and **0.673 macro F1**:

| Sentiment | Precision | Recall | F1 |
|---|---:|---:|---:|
| Negative | 0.638 | 0.630 | 0.634 |
| Neutral | 0.356 | 0.493 | 0.413 |
| Positive | 0.982 | 0.965 | 0.973 |

Because the test set contains many more positive reviews than negative or neutral reviews, accuracy alone is not representative. The neutral class remains the most challenging. Metrics are from the held-out test split and may vary with training settings.

The public model artifacts are hosted at [`Zinaelnahel/review-sentiment-model`](https://huggingface.co/Zinaelnahel/review-sentiment-model). The model weights are not stored in this Git repository.

## Pseudocode walkthrough

### Training (`train_transformer.py`)

```text
Read 1429_1.csv
Map ratings 1-2 to negative, 3 to neutral, and 4-5 to positive
Keep review title/body text; discard missing labels and empty reviews
Create stratified training, validation, and test splits
For each epoch:
    Fine-tune DistilBERT with class-weighted loss
    Measure macro F1 on validation data
    Save the checkpoint if validation macro F1 improves
Evaluate the selected checkpoint on the held-out test data
Save model artifacts and test metrics
```

### Web app (`app.py`)

```text
Load the public tokenizer and classifier once, then cache them
Ask the visitor to enter a review
On submit, tokenize and truncate the review
Run the classifier without gradient tracking
Display the predicted sentiment and probabilities for all three classes
```

### Publish model artifacts (`upload_model_to_hf.py`)

```text
Verify Hugging Face authentication and account ownership of the target namespace
Create or verify a public model repository
Upload only the model, tokenizer, and evaluation files
Print the public model URL used by the app
```

## Train the model

From the repository root:

```bash
python -m pip install -r requirements-train.txt
python train_transformer.py
```

Training downloads the pretrained DistilBERT checkpoint if needed and saves the selected model, tokenizer, and held-out evaluation under `models/distilbert-sentiment/`. A CUDA-capable GPU is recommended; CPU training can be slow.

Optional training arguments:

```bash
python train_transformer.py --epochs 3 --batch-size 16 --max-length 192
```

To publish a newly trained checkpoint, authenticate locally using `hf auth login`, then run:

```bash
python upload_model_to_hf.py
```

The uploader publishes only the model and tokenizer files to the Hugging Face model repository. It does not upload CSV datasets or training code. Never place an access token in source code or chat.

## Run the app locally

Install runtime dependencies and launch Streamlit:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app downloads the model from Hugging Face on first use and caches it for later predictions. To use a different public model repository, set the `SENTIMENT_MODEL_ID` environment variable.

## Deploy on Streamlit Community Cloud

1. Push the application files to a GitHub repository. Keep the model weights out of Git; this app downloads them from the public Hugging Face model repository.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) using the GitHub account that can access the repository.
3. Choose **Create app**, select this repository and its `main` branch, and set the main file path to `app.py`.
4. Deploy. Streamlit installs the dependencies in `requirements.txt`; the first app load downloads the model weights and may take a few minutes.

The app uses the public `Zinaelnahel/review-sentiment-model` model ID by default. If you publish a different model repo, set `SENTIMENT_MODEL_ID` in the app's Streamlit secrets/environment settings.

## Project files

- `models/sentiment_analysis_model.ipynb` — data exploration, baseline, transformer experiment, and evaluation.
- `train_transformer.py` — reproducible training, validation-based checkpoint selection, and held-out test evaluation.
- `app.py` — Streamlit prediction interface with all three class probabilities.
- `upload_model_to_hf.py` — authenticated uploader for model artifacts only.
- `requirements.txt` — lightweight Streamlit inference dependencies.
- `requirements-train.txt` — training and Hugging Face upload dependencies.
- `data/1429_1.csv` — source review dataset used for training.
