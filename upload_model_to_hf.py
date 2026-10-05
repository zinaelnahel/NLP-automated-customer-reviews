"""Upload the trained classifier to a public Hugging Face model repository.

PSEUDOCODE
1. Verify Hugging Face login and that the repository is in the logged-in user's namespace.
2. Create the public model repository if it does not exist.
3. Upload only model, tokenizer, and evaluation artifacts from the checkpoint folder.
4. Print the repository URL used by the Streamlit app.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "distilbert-sentiment"
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
OPTIONAL_FILES = ("special_tokens_map.json", "evaluation.json")


def main() -> None:
    """Upload only the trained model artifacts to the selected public repo."""
    parser = argparse.ArgumentParser(
        description="Publish the sentiment model to a Hugging Face model repository."
    )
    parser.add_argument(
        "--repo-id",
        default="Zinaelnahel/review-sentiment-model",
        help="Model repository ID (default: Zinaelnahel/review-sentiment-model).",
    )
    args = parser.parse_args()
    if len(args.repo_id.split("/")) != 2:
        parser.error("--repo-id must be in namespace/name format.")

    missing = [name for name in REQUIRED_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Model files missing from {MODEL_DIR}: {', '.join(missing)}. "
            "Run `python train_transformer.py` first."
        )

    api = HfApi()
    identity = api.whoami()
    namespace = args.repo_id.split("/", maxsplit=1)[0]
    if identity["name"].casefold() != namespace.casefold():
        raise PermissionError(
            f"Authenticate as {namespace} to upload to {args.repo_id}; "
            f"the current Hugging Face identity is {identity['name']}."
        )
    print(f"Authenticated with Hugging Face as {identity['name']}.")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    repo_info = api.repo_info(args.repo_id, repo_type="model")
    if repo_info.private:
        raise PermissionError(
            f"{args.repo_id} is private. Make it public before deploying the app."
        )

    allowed_files = list(REQUIRED_FILES) + list(OPTIONAL_FILES)
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=args.repo_id,
        repo_type="model",
        allow_patterns=allowed_files,
        commit_message="Upload fine-tuned customer review sentiment model",
    )
    print(f"Public model: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
