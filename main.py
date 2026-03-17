from api import get_comments
from sentiment import predict_sentiment, is_toxic

def analyze_video(video_id):
    comments = get_comments(video_id)

    if not comments:
        print("No comments found ❌")
        return

    positive = 0
    negative = 0
    neutral = 0
    toxic_count = 0

    print("\nSample Comments:\n")

    for i, comment in enumerate(comments):
        sentiment = predict_sentiment(comment)
        toxic = is_toxic(comment)

        # show only first 10 comments
        if i < 10:
            label = sentiment.upper()
            if toxic:
                label += " ☠️"
            print(f"{label} ➜ {comment[:80]}...")

        # count sentiments
        if sentiment == "positive":
            positive += 1
        elif sentiment == "negative":
            negative += 1
        else:
            neutral += 1

        if toxic:
            toxic_count += 1

    total = len(comments)

    print("\n📊 Results:")
    print(f"Total Comments: {total}")
    print(f"Positive: {positive/total * 100:.2f}%")
    print(f"Neutral: {neutral/total * 100:.2f}%")
    print(f"Negative: {negative/total * 100:.2f}%")
    print(f"Toxic Comments: {toxic_count} ({toxic_count/total * 100:.2f}%)")


if __name__ == "__main__":
    while True:
        video_id = input("\nEnter YouTube video ID (or 'exit'): ")

        if video_id.lower() == "exit":
            print("Exiting 👋")
            break

        analyze_video(video_id)