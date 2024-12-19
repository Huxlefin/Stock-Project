import tweepy
from transformers import pipeline
import matplotlib.pyplot as plt
from datetime import datetime

# Twitter API credentials
bearer_token = "AAAAAAAAAAAAAAAAAAAAAJTVvwEAAAAA%2BlXUuDq6cElBt6uCZ8z%2Fb3wx3PQ%3DqhFPJEhjf440xMmrOEtTWweFJ8os3K7ZrGpOI6OSchSx9qnYMM"

# Authenticate client
client = tweepy.Client(bearer_token=bearer_token)

# Initialize Hugging Face sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Search query for tweets about Apple stock, using relevant keywords and hashtags
query = 'AAPL OR "Apple stock" OR "Apple shares" OR #AAPL OR #AppleStock OR "stock market" OR "Apple trading" lang:en'


# Fetch recent tweets (max 10 per request)
response = client.search_recent_tweets(
    query=query,
    max_results=10,  # Adjust this as needed (up to 100 tweets per request)
    tweet_fields=["created_at", "text"]
)

# Initialize dictionaries to count sentiment types by date
date_sentiment_counts = {}

if response.data:
    for tweet in response.data:
        # Extract the date from the tweet's 'created_at' field
        tweet_date = tweet.created_at.date()
        
        # Perform sentiment analysis on the tweet's text
        sentiment = sentiment_analyzer(tweet.text[:512])[0]  # Analyze up to 512 characters
        sentiment_label = sentiment['label']  # Get the sentiment label (POSITIVE, NEGATIVE, NEUTRAL)

        # Initialize the date entry if not already present
        if tweet_date not in date_sentiment_counts:
            date_sentiment_counts[tweet_date] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}

        # Increment the count for the detected sentiment
        date_sentiment_counts[tweet_date][sentiment_label] += 1

    # Prepare data for plotting
    sorted_dates = sorted(date_sentiment_counts.keys())
    positive_counts = [date_sentiment_counts[date]['POSITIVE'] for date in sorted_dates]
    negative_counts = [date_sentiment_counts[date]['NEGATIVE'] for date in sorted_dates]
    neutral_counts = [date_sentiment_counts[date]['NEUTRAL'] for date in sorted_dates]

    # Plotting the data
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(sorted_dates, positive_counts, color='green', alpha=0.6, label='Positive Tweets')
    ax.bar(sorted_dates, negative_counts, color='red', alpha=0.6, label='Negative Tweets', bottom=positive_counts)
    ax.bar(sorted_dates, neutral_counts, color='gray', alpha=0.6, label='Neutral Tweets', 
           bottom=[p + n for p, n in zip(positive_counts, negative_counts)])

    # Adding labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Tweets')
    ax.set_title('Tweet Sentiment Distribution Over Time')
    ax.legend(loc='upper left')

    # Display the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No tweets found or access issue.")
