import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "distilbert-base-uncased"
LABEL_NAMES = ["negative", "neutral", "positive"]
RATING_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}


def load_modeling_data(csv_path: Path) -> pd.DataFrame:
    """Load review text and derive sentiment labels from star ratings.

    PSEUDOCODE: read the CSV -> map star ratings to labels -> combine
    title and body -> discard invalid labels and empty review bodies.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    ratings = pd.to_numeric(df["reviews.rating"], errors="coerce")
    review_text = df["reviews.text"].fillna("").astype(str).str.strip()
    review_title = df["reviews.title"].fillna("").astype(str).str.strip()
    sentiment = ratings.map(RATING_TO_SENTIMENT)
    text = review_text.where(review_title.eq(""), review_title + ". " + review_text)
    valid = sentiment.notna() & review_text.ne("")
    return pd.DataFrame({"text": text[valid], "sentiment": sentiment[valid]}).reset_index(drop=True)
def encode_split(
    tokenizer: AutoTokenizer,
    texts: pd.Series,
    labels: pd.Series,
    label_to_id: dict[str, int],
    max_length: int,
) -> TensorDataset:
    """Tokenize one split and pair token tensors with numeric labels.

    PSEUDOCODE: tokenize/truncate each review -> encode its label ->
    return input IDs, attention masks, and labels as a dataset.
    """
    tokens = tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    label_ids = torch.tensor(labels.map(label_to_id).to_numpy(), dtype=torch.long)
    return TensorDataset(tokens["input_ids"], tokens["attention_mask"], label_ids)


def predict(model, loader: DataLoader, device: torch.device) -> tuple[list[int], float]:
    """Run a model over a split and return predicted classes and mean loss.

    PSEUDOCODE: switch to inference mode -> predict each batch ->
    collect class IDs and loss -> return the predictions and average loss.
    """
    model.eval()
    predictions = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            total_loss += torch.nn.functional.cross_entropy(logits, labels).item()
            batches += 1
    return predictions, total_loss / batches


def main() -> None:
    """Train, select, and evaluate the DistilBERT sentiment classifier.

    PSEUDOCODE: parse training options -> seed and load data -> create
    stratified splits -> fine-tune and validate each epoch -> keep the
    best validation-macro-F1 checkpoint -> evaluate once on the test set.
    """
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate DistilBERT for review sentiment.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "1429_1.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "distilbert-sentiment")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1:
        parser.error("--epochs, --batch-size, and --max-length must be positive")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model_df = load_modeling_data(args.data)
    label_to_id = {label: index for index, label in enumerate(LABEL_NAMES)}
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        model_df["text"],
        model_df["sentiment"],
        test_size=0.1,
        random_state=42,
        stratify=model_df["sentiment"],
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=1 / 9,
        random_state=42,
        stratify=train_labels,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}; examples train/validation/test: "
          f"{len(train_texts):,}/{len(val_texts):,}/{len(test_texts):,}", flush=True)
    if device.type == "cpu":
        print("Warning: fine-tuning on CPU can take a long time.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = encode_split(tokenizer, train_texts, train_labels, label_to_id, args.max_length)
    val_dataset = encode_split(tokenizer, val_texts, val_labels, label_to_id, args.max_length)
    test_dataset = encode_split(tokenizer, test_texts, test_labels, label_to_id, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=device.type == "cuda")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    ).to(device)
    class_counts = np.bincount(
        train_labels.map(label_to_id).to_numpy(),
        minlength=len(LABEL_NAMES),
    )
    weights = len(train_labels) / (len(LABEL_NAMES) * class_counts)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    args.output.mkdir(parents=True, exist_ok=True)
    best_val_macro_f1 = -1.0
    best_epoch = 0

    # Fine-tune each epoch, validate on the held-out validation split, and
    # save only when validation macro F1 improves.
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", mininterval=5)
        for input_ids, attention_mask, labels in progress:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.3f}")

        val_predictions, val_loss = predict(model, val_loader, device)
        val_truth = val_labels.map(label_to_id).to_numpy()
        val_macro_f1 = f1_score(
            val_truth,
            val_predictions,
            labels=list(range(len(LABEL_NAMES))),
            average="macro",
            zero_division=0,
        )
        mean_train_loss = running_loss / len(train_loader)
        print(
            f"Epoch {epoch}: train_loss={mean_train_loss:.4f}, "
            f"validation_loss={val_loss:.4f}, validation_macro_f1={val_macro_f1:.4f}",
            flush=True,
        )
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)

    # Keep the test split out of model selection; evaluate it only after
    # the best checkpoint has already been selected.
    best_model = AutoModelForSequenceClassification.from_pretrained(args.output).to(device)
    test_predictions, _ = predict(best_model, test_loader, device)
    test_truth = test_labels.map(label_to_id).to_numpy()
    report = classification_report(
        test_truth,
        test_predictions,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_truth, test_predictions, labels=list(range(len(LABEL_NAMES))))
    metrics = {
        "model": MODEL_NAME,
        "best_epoch": best_epoch,
        "epochs_trained": args.epochs,
        "validation_macro_f1": best_val_macro_f1,
        "test_accuracy": accuracy_score(test_truth, test_predictions),
        "test_classification_report": report,
        "test_confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": LABEL_NAMES,
        "data_rows": len(model_df),
        "split_sizes": {
            "train": len(train_texts),
            "validation": len(val_texts),
            "test": len(test_texts),
        },
        "rating_to_sentiment": {str(key): value for key, value in RATING_TO_SENTIMENT.items()},
        "max_length": args.max_length,
    }
    (args.output / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Best epoch: {best_epoch}; test accuracy: {metrics['test_accuracy']:.2%}")
    print(pd.DataFrame(report).T.to_string(float_format=lambda value: f"{value:.3f}"))
    print("Test confusion matrix (rows=actual, columns=predicted):")
    print(pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES))
    print(f"Saved model and evaluation to {args.output}")


if __name__ == "__main__":
    main()
