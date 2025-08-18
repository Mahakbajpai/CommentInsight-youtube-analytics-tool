# 1. Imports
import pandas as pd
from googleapiclient.discovery import build
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# 2. YouTube API Setup
# Replace "YOUR_YOUTUBE_API_KEY" with your actual API key
API_KEY = "YOUR_YOUTUBE_API_KEY"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

# Function to extract video ID from URL
def extract_video_id(youtube_url):
    """
    Extracts the video ID from any standard YouTube URL format.
    """
    match = re.search(r"(?:v=|/)([a-zA-Z0-9_-]{11})", youtube_url)
    if match:
        return match.group(1)
    return None

# Function to get comments from a video
def get_youtube_comments(video_id):
    comments = []
    next_page_token = None
    
    # Get comment threads (top-level comments)
    while True:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
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

# 3. Text Pre-processing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# 4. Sentiment Analysis Function
def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment, polarity

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
            df[['sentiment', 'polarity']] = df['cleaned_comment'].apply(lambda x: pd.Series(analyze_sentiment(x)))

            # Display sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            print("\nSentiment Analysis Results:")
            print(sentiment_counts)
            
            # --- Visualizations ---
            
            # 1. Bar Plot of Sentiment Counts
            plt.figure(figsize=(8, 6))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
            plt.title('Distribution of YouTube Comment Sentiments')
            plt.xlabel('Sentiment')
            plt.ylabel('Number of Comments')
            plt.show()

            # 2. Pie Chart of Sentiment Proportions
            plt.figure(figsize=(8, 8))
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(sentiment_counts)))
            plt.title('Sentiment Proportions in YouTube Comments')
            plt.show()

            # 3. Word Cloud
            all_comments_text = ' '.join(df['cleaned_comment'])
            if all_comments_text:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments_text)
                plt.figure(figsize=(10, 7))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Most Frequent Words in Comments')
                plt.show()
            
            # 4. Histogram of Sentiment Scores
            plt.figure(figsize=(10, 6))
            sns.histplot(df['polarity'], bins=20, kde=True, color='skyblue')
            plt.title('Distribution of Sentiment Polarity Scores')
            plt.xlabel('Polarity Score (-1 to +1)')
            plt.ylabel('Number of Comments')
            plt.axvline(x=0, color='red', linestyle='--', label='Neutrality line')
            plt.legend()
            plt.show()

        else:
            print("No comments found or an error occurred.")
    else:
        print("Invalid YouTube URL provided.")
