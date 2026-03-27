from textblob import TextBlob
import re

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return text.strip()

def predict_sentiment(text):
    text = clean_text(text)
    if not text:
        return "neutral"
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
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