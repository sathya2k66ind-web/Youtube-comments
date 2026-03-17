from googleapiclient.discovery import build

# 🔑 Your API key (MAKE SURE it's in quotes)
API_KEY = "AIzaSyC4P1WHROvLc-nNwMUnfpx4tqj5altIHpg"

# 🔧 Initialize YouTube API
youtube = build("youtube", "v3", developerKey=API_KEY)


# 🎥 Get video details (title, channel, description)
def get_video_details(video_id):
    try:
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()

        if not response["items"]:
            return None

        data = response["items"][0]["snippet"]

        return {
            "title": data["title"],
            "channel": data["channelTitle"],
            "description": data["description"][:300]  # limit length
        }

    except Exception as e:
        print("Error fetching video details:", e)
        return None


# 💬 Get comments
def get_comments(video_id, max_comments=50):
    comments = []

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )

        response = request.execute()

        while request and len(comments) < max_comments:
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

                if len(comments) >= max_comments:
                    break

            # get next page
            request = youtube.commentThreads().list_next(request, response)

            if request:
                response = request.execute()

    except Exception as e:
        print("Error fetching comments:", e)

    return comments