"""Streamlit frontend for the public DistilBERT sentiment model.

PSEUDOCODE
1. Load the tokenizer and classifier once from the public model repository.
2. Show an input box for a customer review.
3. On button click, tokenize the text and get three class probabilities.
4. Display the predicted class and each class's confidence score.
"""

import os

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = os.getenv("SENTIMENT_MODEL_ID", "Zinaelnahel/review-sentiment-model")
MAX_LENGTH = 192
SENTIMENT_STYLE = {
    "positive": {"emoji": "😊", "color": "#047857"},
    "neutral": {"emoji": "😐", "color": "#b45309"},
    "negative": {"emoji": "🙁", "color": "#be123c"},
}

APP_CSS = """
.stApp {
    background:
        radial-gradient(ellipse at 8% 0%, rgba(129, 140, 248, 0.18), transparent 34rem),
        radial-gradient(ellipse at 100% 14%, rgba(45, 212, 191, 0.14), transparent 32rem),
        #f6f7fb;
}
[data-testid="stHeader"] { background: rgba(246, 247, 251, 0.82); }
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 900px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}
.hero {
    padding: 2rem 2.2rem;
    border: 1px solid rgba(255, 255, 255, 0.7);
    border-radius: 26px;
    background: linear-gradient(125deg, #312e81 0%, #4f46e5 56%, #0f766e 125%);
    box-shadow: 0 18px 50px rgba(49, 46, 129, 0.2);
    color: white;
    margin-bottom: 1.4rem;
}
.hero-kicker {
    margin: 0 0 0.65rem 0;
    color: #c7d2fe;
    font-size: 0.78rem;
    font-weight: 750;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}
.hero h1 {
    margin: 0;
    color: white;
    font-size: clamp(2rem, 5vw, 3rem);
    line-height: 1.1;
}
.hero-copy {
    max-width: 610px;
    margin: 0.8rem 0 0 0;
    color: #e0e7ff;
    font-size: 1.05rem;
    line-height: 1.6;
}
.section-title {
    margin: 1.2rem 0 0.35rem 0;
    color: #1e1b4b;
    font-size: 1.3rem;
    font-weight: 750;
}
.section-copy { margin: 0 0 0.7rem 0; color: #64748b; }
.result-card {
    padding: 1.3rem 1.5rem;
    border: 1px solid #e0e7ff;
    border-radius: 20px;
    background: linear-gradient(135deg, #ffffff, #eef2ff);
    box-shadow: 0 12px 34px rgba(49, 46, 129, 0.09);
}
.result-label {
    margin: 0 0 0.2rem 0;
    color: #64748b;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.result-sentiment {
    margin: 0;
    color: #1e1b4b;
    font-size: 1.8rem;
    font-weight: 800;
}
.result-confidence { margin: 0.55rem 0 0 0; color: #475569; }
.legend-card {
    height: 100%;
    padding: 0.9rem 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.92);
}
.legend-name { margin: 0; font-weight: 750; }
.legend-note { margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.86rem; }
div.stButton > button[kind="primary"] {
    min-height: 3rem;
    border: 0;
    border-radius: 13px;
    background: linear-gradient(100deg, #4f46e5, #0f766e);
    color: white;
    font-weight: 750;
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.2);
}
div.stButton > button[kind="primary"]:hover {
    border: 0;
    background: linear-gradient(100deg, #4338ca, #0f766e);
    color: white;
}
[data-testid="stTextArea"] textarea {
    border-radius: 14px;
    border-color: #c7d2fe;
    background: #ffffff;
    color: #1e293b;
}
[data-testid="stTextArea"] textarea::placeholder { color: #64748b; opacity: 1; }
"""


@st.cache_resource
def load_sentiment_model():
    """Load the public tokenizer and model once per Streamlit server.

    PSEUDOCODE: read the configured model ID -> download/cache tokenizer
    and weights -> choose CPU/GPU -> switch the classifier to inference mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_sentiment(review: str) -> tuple[str, float, dict[str, float]]:
    """Predict one review and return its class and all class probabilities.

    PSEUDOCODE: reject empty review -> tokenize/truncate -> run inference
    without gradients -> convert logits to probabilities -> return top label,
    its confidence, and the complete class-score dictionary.
    """
    if not review or not review.strip():
        raise ValueError("Enter a review before requesting a prediction.")

    tokenizer, model, device = load_sentiment_model()
    encoded = tokenizer(
        review.strip(),
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=0).cpu().tolist()

    scores = {}
    for index, probability in enumerate(probabilities):
        label = model.config.id2label[index].lower()
        if label not in SENTIMENT_STYLE:
            raise ValueError(f"Unexpected sentiment label in model: {label}")
        scores[label] = float(probability)

    predicted_label = max(scores, key=scores.get)
    return predicted_label, scores[predicted_label], scores


# PSEUDOCODE: set page theme -> show a colorful header and review box
# -> classify on button click -> show result and the three class scores.
st.set_page_config(page_title="Review Sentiment Classifier", page_icon="💬")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
        <p class="hero-kicker">✨ AI-powered review insights</p>
        <h1>How do customers feel?</h1>
        <p class="hero-copy">
            Turn a product review into a quick sentiment snapshot.
            Paste the words, and let the model read between the lines.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="section-title">📝 Try a review</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Add review text below. We’ll classify its sentiment.</p>',
    unsafe_allow_html=True,
)
review_text = st.text_area(
    "Review text",
    height=170,
    placeholder="Example: Setup was quick, and the sound quality is fantastic!",
    label_visibility="collapsed",
)
st.caption(f"✍️ {len(review_text.strip()):,} characters")

if st.button("🔎  Analyze sentiment", type="primary", use_container_width=True):
    if not review_text.strip():
        st.warning("📝 Add some review text first, then try again.")
    else:
        try:
            label, confidence, scores = predict_sentiment(review_text)
        except ValueError as error:
            st.error(str(error))
        else:
            style = SENTIMENT_STYLE[label]
            st.markdown(
                f"""
                <div class="result-card">
                    <p class="result-label">💡 Sentiment detected</p>
                    <p class="result-sentiment">
                        {style["emoji"]}
                        <span style="color:{style["color"]}">{label.capitalize()}</span>
                    </p>
                    <p class="result-confidence">
                        🎯 Model confidence: <strong>{confidence:.1%}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="section-title">📊 Class probabilities</p>',
                unsafe_allow_html=True,
            )
            probability_columns = st.columns(3)
            for column, class_label in zip(
                probability_columns,
                ("negative", "neutral", "positive"),
            ):
                class_style = SENTIMENT_STYLE[class_label]
                probability = scores[class_label]
                with column:
                    st.markdown(
                        f"""
                        <div class="legend-card">
                            <p class="legend-name" style="color:{class_style["color"]}">
                                {class_style["emoji"]} {class_label.capitalize()}
                            </p>
                            <p class="legend-note">{probability:.1%} probability</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(probability)
            st.caption("ℹ️ Confidence is the model’s probability for its prediction, not a guarantee.")

if not review_text.strip():
    st.markdown('<p class="section-title">🌈 The three sentiments</p>', unsafe_allow_html=True)
    legend_columns = st.columns(3)
    legend_items = [
        ("😊 Positive", "Happy, approving, or satisfied"),
        ("😐 Neutral", "Mixed, factual, or undecided"),
        ("🙁 Negative", "Unhappy, critical, or disappointed"),
    ]
    for column, (name, description) in zip(legend_columns, legend_items):
        with column:
            st.markdown(
                f"""
                <div class="legend-card">
                    <p class="legend-name">{name}</p>
                    <p class="legend-note">{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
