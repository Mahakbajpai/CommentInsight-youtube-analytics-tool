# YouTube Comments Sentiment Analysis

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from textblob import TextBlob

# --------------------------
# CONFIG
# --------------------------
API_KEY = "AIzaSyCqXA4gBKHLPJwLpUSyRv6zbQ8dNFgLueQ"   # replace with your YouTube Data API key
VIDEO_ID = "K7x8W06VjZY" # replace with YouTube video ID
MAX_COMMENTS = 200

# --------------------------
# Function to fetch comments using API
# --------------------------
def get_youtube_comments(api_key, video_id, max_results=200):
    comments = []
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": api_key,
        "maxResults": 100,
        "textFormat": "plainText"
    }

    while len(comments) < max_results:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print("❌ API Error:", response.json())
            break

        data = response.json()
        for item in data["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_results:
                break

        # pagination
        if "nextPageToken" in data:
            params["pageToken"] = data["nextPageToken"]
        else:
            break

    return comments

# --------------------------
# Load Data
# --------------------------
try:
    comments = get_youtube_comments(API_KEY, VIDEO_ID, MAX_COMMENTS)
    df = pd.DataFrame(comments, columns=["comment"])
    print(f"✅ Downloaded {len(df)} comments")
except:
    print("⚠️ Using sample fallback data...")
    sample_comments = [
        "I love this video!", "This is terrible...", "Amazing content, thanks!",
        "Not what I expected.", "So helpful, I learned a lot!", "Waste of time."
    ]
    df = pd.DataFrame(sample_comments, columns=["comment"])

# --------------------------
# Cleaning
# --------------------------
df["comment"] = df["comment"].str.lower()

# --------------------------
# Sentiment Analysis
# --------------------------
df["polarity"] = df["comment"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["sentiment"] = df["polarity"].apply(
    lambda x: "positive" if x > 0 else ("negative" if x < 0 else "neutral")
)

# --------------------------
# EDA & Plots
# --------------------------
print(df.head())

# Sentiment distribution
sentiment_counts = df["sentiment"].value_counts()

plt.figure(figsize=(6,4))
sentiment_counts.plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Polarity histogram
plt.figure(figsize=(6,4))
df["polarity"].plot(kind="hist", bins=20, edgecolor="black")
plt.title("Polarity Distribution")
plt.show()

# --------------------------
# Insights
# --------------------------
print("\n📌 Insights:")
print(f"1. Total comments analyzed: {len(df)}")
print(f"2. Positive comments: {sum(df['sentiment']=='positive')}")
print(f"3. Negative comments: {sum(df['sentiment']=='negative')}")
print(f"4. Neutral comments: {sum(df['sentiment']=='neutral')}")
print(f"5. Average polarity: {df['polarity'].mean():.2f}")
