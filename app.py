from flask import Flask, render_template, request
from api import get_comments, get_video_details
from sentiment import predict_sentiment, is_toxic
 
app = Flask(__name__)
 
# ── Landing page ──────────────────────────────────────
@app.route("/landing")
def landing():
    return render_template("landing.html")
 
# ── Main analyzer ─────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    results       = None
    comments_data = []
    video_data    = None
    video_id      = None
 
    if request.method == "POST":
        video_url = request.form["video_url"]
        video_id  = video_url.split("v=")[-1]
 
        comments   = get_comments(video_id)
        video_data = get_video_details(video_id)
 
        positive = negative = neutral = toxic_count = 0
 
        for comment in comments:
            sentiment = predict_sentiment(comment)
            toxic     = is_toxic(comment)
 
            comments_data.append({
                "text":      comment,
                "sentiment": sentiment,
                "toxic":     toxic
            })
 
            if sentiment == "positive":   positive += 1
            elif sentiment == "negative": negative += 1
            else:                         neutral  += 1
 
            if toxic: toxic_count += 1
 
        total = len(comments)
 
        results = {
            "positive": round(positive    / total * 100, 2),
            "neutral":  round(neutral     / total * 100, 2),
            "negative": round(negative    / total * 100, 2),
            "toxic":    round(toxic_count / total * 100, 2)
        }
 
    return render_template("index.html",
                           results=results,
                           comments=comments_data[:10],
                           video=video_data,
                           video_id=video_id)
 
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
 