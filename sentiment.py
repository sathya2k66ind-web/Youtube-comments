from transformers import pipeline
import re

# ── Load model once ──────────────────────────────────────
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    max_length=512,
    truncation=True
)

# ── Gen Z slang dictionaries ─────────────────────────────
POSITIVE_SLANG = {
    # hype / approval
    "w", "W", "goat", "GOAT", "slay", "slayed", "slaying", "ate", "no crumbs",
    "based", "bussin", "bussing", "fire", "🔥", "lit", "goated", "king", "queen",
    "iconic", "legend", "legendary", "valid", "facts", "fax", "real", "realest",
    "clean", "hard", "hits hard", "banger", "certified", "lowkey love", "ngl love",
    "underrated", "deserved", "earned", "respect", "🙌", "💯", "❤️", "😍", "🤩",
    "🥹", "😭❤️", "crying laughing", "imo goat", "no cap", "periodt", "period",
    "understood the assignment", "ate that", "giving", "its giving", "snatched",
    "rent free", "im dead 💀", "lmaoo", "bestie", "sheesh", "lets go", "let's go",
    "lesgo", "lets goo", "pog", "poggers", "gz", "gg", "well played", "carry",
    "absolute unit", "big W", "massive W", "this is peak", "peak content",
    "peak", "peak cinema", "cinema", "10/10", "100/100", "perfect", "flawless",
}

NEGATIVE_SLANG = {
    # L takes / disapproval
    "L", "big L", "massive L", "ratio", "ratioed", "mid", "meh", "nah",
    "trash", "garbage", "dog water", "washed", "cooked", "fell off", "fell off hard",
    "flop", "flopped", "flops", "overrated", "cringe", "cringy", "yikes",
    "bruh", "smh", "not it", "not him", "not her", "not them", "sus",
    "cap", "capping", "liar", "fake", "clickbait", "scam", "fraud",
    "boring", "mid at best", "disappointed", "disappointment", "waste of time",
    "skip", "skip this", "nobody asked", "who asked", "nobody cares",
    "clout chasing", "clout", "attention seeker", "pick me", "npc",
    "💀 them", "they fumbled", "fumbled", "threw", "choker", "choked",
    "🗑️", "🤮", "🤢", "😒", "😤", "🙄", "💔", "😞",
}

TOXIC_SLANG = {
    # explicit toxicity
    "hate", "stupid", "idiot", "worst", "trash", "ugly", "dumb",
    "kill", "die", "garbage", "pathetic", "loser", "scum", "awful",
    "disgusting", "moron", "retard", "racist", "nazi", "stfu", "shut up",
    "kys", "neck yourself", "ratio + L + no", "cope", "cope harder",
    "seethe", "malding", "virgin", "incel", "bot", "braindead",
    "skill issue", "go outside", "touch grass", "delusional", "clown",
    "🤡", "💩",
}

# ── Preprocessing ─────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return text.strip()

def extract_tokens(text):
    """Return lowercased word tokens + keep emojis"""
    tokens = set(re.findall(r"[\w']+|[^\w\s]", text.lower()))
    return tokens

# ── Slang override check ──────────────────────────────────
def slang_check(text):
    """
    Returns 'positive', 'negative', 'neutral', or None.
    None means no strong slang signal found — fall through to model.
    """
    tokens = extract_tokens(text)
    raw_lower = text.lower()

    pos_hits = sum(1 for s in POSITIVE_SLANG if s.lower() in raw_lower or s in tokens)
    neg_hits = sum(1 for s in NEGATIVE_SLANG if s.lower() in raw_lower or s in tokens)

    # Strong single signal
    if pos_hits >= 1 and neg_hits == 0:
        return 'positive'
    if neg_hits >= 1 and pos_hits == 0:
        return 'negative'

    # Tie or mixed — fall through to model
    return None

# ── Short comment handler ─────────────────────────────────
def handle_short(text):
    """
    For very short comments (under 4 words) the model is unreliable.
    Use slang dict + simple heuristics instead.
    """
    lower = text.lower().strip()

    # Pure positive emojis/words
    pure_pos = {"❤️","😍","🤩","🙌","💯","🥹","fire","W","goat","slay","king","queen","banger","facts","based","lit","iconic","pog","gg"}
    # Pure negative
    pure_neg = {"L","💩","🤮","🤢","😒","🙄","💔","🗑️","trash","mid","ratio","yikes","smh","cringe","boring","skip"}

    for p in pure_pos:
        if p.lower() in lower:
            return 'positive'
    for n in pure_neg:
        if n.lower() in lower:
            return 'negative'

    return None

# ── Main predict function ─────────────────────────────────
def predict_sentiment(text):
    text = clean_text(text)
    if not text:
        return "neutral"

    word_count = len(text.split())

    # Short comment path
    if word_count <= 3:
        result = handle_short(text)
        if result:
            return result
        # Still short but no signal — run model anyway
        try:
            r = sentiment_pipeline(text)[0]
            label = r['label'].lower()
            score = r['score']
            # Only trust model on short text if confidence is high
            if score >= 0.75:
                return label
            else:
                return 'neutral'
        except Exception:
            return 'neutral'

    # Slang override for longer text
    slang_result = slang_check(text)
    if slang_result:
        return slang_result

    # Standard model path
    try:
        r = sentiment_pipeline(text)[0]
        label = r['label'].lower()
        score = r['score']

        # Raise the bar for neutral — only call it neutral if model is confident
        if label == 'neutral' and score < 0.65:
            # Low confidence neutral — try to resolve with slang
            if any(s.lower() in text.lower() for s in POSITIVE_SLANG):
                return 'positive'
            if any(s.lower() in text.lower() for s in NEGATIVE_SLANG):
                return 'negative'

        return label

    except Exception:
        return 'neutral'


# ── Toxicity check ────────────────────────────────────────
def is_toxic(text):
    lower = text.lower()
    return any(word.lower() in lower for word in TOXIC_SLANG)