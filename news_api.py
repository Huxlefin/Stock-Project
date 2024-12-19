import pandas as pd
from datetime import datetime
import requests
import time
from transformers import pipeline
import matplotlib.pyplot as plt

data = pd.read_csv("sp500LastYear.csv")
unique_symbols = data['Symbol'].unique()

# Load the sentiment-analysis pipeline
classifier = pipeline('sentiment-analysis')

# Finnhub API key (replace with your own)
API_KEY = 'csnui6hr01qkfk595rn0csnui6hr01qkfk595rng'

for stock in unique_symbols:
    stock_data = data[data['Symbol'] == stock]
    stock_data = stock_data.dropna(subset=['Close'])

    for i in range(len(stock_data) - 1):  # Loop until the second-to-last element
        stock1 = stock_data.iloc[i]  # Get the row at index i
        stock2 = stock_data.iloc[i + 1]  # Get the next row at index i+1

        # Calculate percentage change in 'Close' price
        pct_change = ((stock2['Close'] - stock1['Close']) / stock1['Close']) * 100

        if pct_change > 3 or pct_change < -3:

            # Convert the date string to a datetime object
            date = datetime.strptime(stock2['Date'], "%Y-%m-%d")
            date_str = stock2['Date']

            # Finnhub API endpoint for company news
            url = f'https://finnhub.io/api/v1/company-news?symbol={stock}&from={date_str}&to={date_str}&token={API_KEY}'

             # Make a GET request to the API
            response = requests.get(url)

            print(f"{stock} changed by {pct_change:.2f}% on {date}")

            # Check if the request was successful
            if response.status_code == 200:
                news_data = response.json()
                
                # Print out the news articles
                if news_data:
                    print(f"News articles for {stock} on {date}:\n")
                    # Perform sentiment analysis on each article
                    sentiments = []
                    for article in news_data:
                        # Combine headline and summary to provide full context
                        text = article['headline'] + " " + article['summary']
                        sentiment = classifier(text, truncation=True, padding=True, max_length=512)
                        sentiments.append((article['headline'], sentiment[0]))

                    # Print the results
                    for headline, sentiment in sentiments:
                        print(f"Headline: {headline}\nSentiment: {sentiment['label']}, Score: {sentiment['score']}\n")

                    positive = sum(1 for sentiment in sentiments if sentiment[1]['label'] == 'POSITIVE')
                    negative = sum(1 for sentiment in sentiments if sentiment[1]['label'] == 'NEGATIVE')
                    neutral = len(sentiments) - positive - negative

                    print(f"Positive Articles: {positive}")
                    print(f"Negative Articles: {negative}")
                    print(f"Neutral Articles: {neutral}")

                    # Prepare data for plotting
                    labels = ['Positive', 'Negative', 'Neutral']
                    values = [positive, negative, neutral]

                    plt.bar(labels, values, color=['green', 'red', 'blue'])
                    plt.title(f"Sentiment Analysis of {stock} News Articles")
                    plt.xlabel("Sentiment")
                    plt.ylabel("Number of Articles")
                    plt.show()

                    total_articles = len(sentiments)
                    positive_percent = (positive / total_articles) * 100
                    negative_percent = (negative / total_articles) * 100
                    neutral_percent = (neutral / total_articles) * 100

                    print(f"Positive Sentiment: {positive_percent:.2f}%")
                    print(f"Negative Sentiment: {negative_percent:.2f}%")
                    print(f"Neutral Sentiment: {neutral_percent:.2f}%")

                    # Define thresholds
                    positive_threshold = 60  # Example threshold for buy signal
                    negative_threshold = 60  # Example threshold for sell signal

                    # Determine buy/hold/sell recommendation
                    if positive_percent >= positive_threshold:
                        recommendation = "BUY"
                    elif negative_percent >= negative_threshold:
                        recommendation = "SELL"
                    else:
                        recommendation = "HOLD"

                    print(f"Recommendation: {recommendation}")
                else:
                    print(f"No news articles found for {stock} on {date}.")
            else:
                print(f"Error {response.status_code}: Unable to retrieve news.")

            # for the request restriction of the api
            time.sleep(2)
