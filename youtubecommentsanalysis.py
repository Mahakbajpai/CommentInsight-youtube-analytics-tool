# 1. Imports
import pandas as pd
from googleapiclient.discovery import build
import re
from textblob import TextBlob # Or from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# 2. YouTube API Setup
# Replace "YOUR_YOUTUBE_API_KEY" with your actual API key
API_KEY = "AIzaSyCqXA4gBKHLPJwLpUSyRv6zbQ8dNFgLueQ"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

# Function to extract video ID from URL
def extract_video_id(youtube_url):
    """
    Extracts the video ID from any standard YouTube URL format.
    """
    # This pattern finds the 11-character video ID by first matching 'v=' or a '/'
    # and then capturing the characters that follow.
    match = re.search(r"(?:v=|/)([a-zA-Z0-9_-]{11})", youtube_url)
    if match:
        return match.group(1)
    return None

# 3. Function to get comments from a video
def get_youtube_comments(video_id):
    comments = []
    next_page_token = None

    while True:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100, # Max results per page
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            print(f"An API error occurred: {e}")
            break
    return comments

# 4. Text Pre-processing
def preprocess_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'@\S+', '', text) # Remove mentions
    text = re.sub(r'#\S+', '', text) # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    return text

# 5. Sentiment Analysis Function
def analyze_sentiment(comment):
    # Using TextBlob
    analysis = TextBlob(comment)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Main execution flow
if __name__ == "__main__":
    youtube_video_url = input("Enter YouTube video URL: ")
    video_id = extract_video_id(youtube_video_url)

    if video_id:
        print(f"Fetching comments for video ID: {video_id}...")
        comments = get_youtube_comments(video_id)
        if comments:
            print(f"Fetched {len(comments)} comments.")

            # Create a DataFrame
            df = pd.DataFrame({'comment': comments})

            # Preprocess comments
            df['cleaned_comment'] = df['comment'].apply(preprocess_text)

            # Analyze sentiment
            df['sentiment'] = df['cleaned_comment'].apply(analyze_sentiment)

            # Display sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            print("\nSentiment Analysis Results:")
            print(sentiment_counts)

            # Visualize the results
            plt.figure(figsize=(8, 6))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
            plt.title('Distribution of YouTube Comment Sentiments')
            plt.xlabel('Sentiment')
            plt.ylabel('Number of Comments')
            plt.show()

        else:
            print("No comments found or an error occurred.")
    else:
        print("Invalid YouTube URL provided.")
