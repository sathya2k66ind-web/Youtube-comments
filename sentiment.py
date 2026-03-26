from transformers import pipeline
import re

# Load once at startup — not on every call
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    max_length=512,
    truncation=True
)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return text.strip()

def predict_sentiment(text):
    text = clean_text(text)
    if not text:
        return "neutral"
    try:
        result = sentiment_pipeline(text)[0]
        label = result['label'].lower()
        # Model returns: positive, neutral, negative
        if label == "positive":
            return "positive"
        elif label == "negative":
            return "negative"
        else:
            return "neutral"
    except Exception:
        return "neutral"

def is_toxic(text):
    toxic_words = [
        "hate", "stupid", "idiot", "worst", "trash", "ugly", "dumb",
        "kill", "die", "garbage", "pathetic", "loser", "scum", "awful",
        "disgusting", "moron", "retard", "racist", "nazi"
    ]
    text = text.lower()
    return any(word in text for word in toxic_words)
```

---

**Two things to update in `requirements.txt` as well**

Remove `textblob` and make sure these are in there —
```
transformers>=4.30.0
torch>=2.0.0