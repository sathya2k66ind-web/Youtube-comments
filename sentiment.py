from textblob import TextBlob
import re

# 🔥 Clean text for better accuracy
def clean_text(text):
    text = re.sub(r"http\S+", "", text)        # remove links
    text = re.sub(r"<.*?>", "", text)          # remove HTML
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove emojis/symbols
    return text.lower()


def predict_sentiment(text):
    text = clean_text(text)  # apply cleaning
    analysis = TextBlob(text)
    score = analysis.sentiment.polarity

    # 🔥 3-class classification
    if score > 0.15:
     return "positive"
    elif score < -0.15:
     return "negative"
    else:
     return "neutral"


def is_toxic(text):
    toxic_words = ["hate", "stupid", "idiot", "worst", "trash", "ugly", "dumb"]

    text = text.lower()
    return any(word in text for word in toxic_words)