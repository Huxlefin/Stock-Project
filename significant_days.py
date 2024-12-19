import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from transformers import pipeline
import requests

data1 = pd.read_csv("sp500LastYear.csv") # only from last year for news purposes
unique_symbols = data1['Symbol'].unique()

# Load the sentiment-analysis pipeline
classifier = pipeline('sentiment-analysis')

# Initialize scalers for X and Y
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Finnhub API key (replace with your own)
API_KEY = 'csnui6hr01qkfk595rn0csnui6hr01qkfk595rng'

plt.figure(figsize = (10, 7))
recommendations = [] # for the graph later on

# plotting the predictions for the data
def plotPrediction(trainData,
                   trainLabels,
                   testData,
                   testLabels,
                   symbol,
                   predictions = None,
                   recommendations=None,
                   future_preds=None,
                   future_dates_seconds=None):

    # plot training data in blue
    plt.scatter(trainData, trainLabels, c="b", s=4, label = "Training Data")

    # plot test data in green
    plt.scatter(testData, testLabels, c='g', s=4, label = "Testing Data")

    if predictions is not None:
        # plot the predictions in red (predictions were made on the test data)
        plt.scatter(testData, predictions, c='r', s=4, label="Predictions")

    # Filter data for the last year
    last_year_start = stock_data1['Date'].min()
    last_year_end = stock_data1['Date'].max()
    last_year_start_epoch = int(pd.Timestamp(last_year_start).timestamp())
    last_year_end_epoch = int(pd.Timestamp(last_year_end).timestamp())
    
    mask = (testData.flatten() >= last_year_start_epoch) & (testData.flatten() <= last_year_end_epoch)
    testData_last_year = testData[mask]
    testLabels_last_year = testLabels[mask]

    # Add recommendations only for the last year
    if recommendations is not None:
        for i, recommendation in enumerate(recommendations[-len(testData_last_year):]):
            plt.annotate(
                recommendation,
                xy=(testData_last_year[i], testLabels_last_year[i]),
                xytext=(testData_last_year[i], testLabels_last_year[i] + 4),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1),
                fontsize=10,
                color='black',
                ha='center',
                va='bottom'
            )
    
    if future_preds is not None:
        plt.scatter(future_dates_seconds, future_preds, c='purple', s=4, label='Future Predictions')

    plt.title(symbol)
    plt.legend(prop={"size": 14})
    plt.show()

# neural network model for predicting the stocks
class StockModel(nn.Module):
    def __init__(self, input_size):
        super(StockModel, self).__init__()
        # neural network layers
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 16)
        self.layer4 = nn.Linear(16, 1) # output layer for linear regression

    # use ReLu function for forward progression in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    

# load and train data for each stock
for stock in unique_symbols:

    # data only from the last year
    stock_data1 = data1[data1['Symbol'] == stock]
    stock_data1 = stock_data1.dropna(subset=['Close'])

    # x-values are the date
    stock_data1['Date'] = pd.to_datetime(data1['Date'])
    timestamps = stock_data1['Date'].astype(np.int64) // 10**9 # convert to seconds since epoch
    X = timestamps.to_numpy().reshape(-1, 1)  # Reshape for PyTorch

    # y-values are the values of the stocks
    Y = stock_data1['Close'].to_numpy().reshape(-1, 1)  # Reshape for PyTorch

    # for news algorithm
    maxY = max(Y)
    minY = min(Y)

    # Scale features and labels
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    Y_train = scaler_Y.fit_transform(Y_train)
    Y_test = scaler_Y.transform(Y_test)

    # convert into pytorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    Y_train_tensor = torch.FloatTensor(Y_train)
    Y_test_tensor = torch.FloatTensor(Y_test)

     # Extrapolate to future dates (e.g., next 5 days)
    last_date = stock_data1['Date'].iloc[-1]
    last_date = pd.to_datetime(last_date, unit='s')

    # Create future dates as list of seconds since epoch
    future_dates = [last_date + timedelta(days=i) for i in range(10, 15)]
    future_dates_seconds = np.array([int(date.timestamp()) for date in future_dates]).reshape(-1, 1)

    # Scale the future dates
    future_dates_scaled = scaler_X.transform(future_dates_seconds)

    # Convert to PyTorch tensor
    future_dates_scaled_tensor = torch.FloatTensor(future_dates_scaled)

    model = StockModel(input_size=X_train.shape[1]) # use model created earlier
    loss = nn.MSELoss() # best loss function for stocks
    optimizer = optim.Adam(model.parameters(), lr=0.01) # best to start with Adam optimizer good for wide range of functions

    # Train the model
    for epoch in range(2000): # number of epochs
        # Put model in training mode (this is the default state of the model)
        model.train();

        # Forward pass on train data using the forward() method inside
        y_preds = model(X_train_tensor)

        # Calculate the loss of the function
        test_loss = loss(y_preds, Y_train_tensor)

        # Zero gradient of the optimizer
        optimizer.zero_grad()

        # Loss backwards
        test_loss.backward()

        # Progress the optimizer
        optimizer.step()
        
        ### Testing
        # evaluate the model
        model.eval()

        with torch.inference_mode():
            # calculate the loss on test data
            test_pred = model(X_test_tensor)
            test_loss = loss(test_pred, Y_test_tensor.type(torch.float))

            # print whats happening every 10 epochs
            if epoch % 100 == 0:
                print(f"Test Loss for {stock}: {test_loss.item()}")

    # Make predictions
    with torch.inference_mode():
        future_predictions = model(future_dates_scaled_tensor)

    with torch.inference_mode():
        y_preds = model(X_test_tensor).numpy()
        y_preds_original = scaler_Y.inverse_transform(y_preds)
        Y_train_original = scaler_Y.inverse_transform(Y_train_tensor.numpy())
        Y_test_original = scaler_Y.inverse_transform(Y_test_tensor.numpy())
        X_train_original = scaler_X.inverse_transform(X_train_tensor.numpy())
        X_test_original = scaler_X.inverse_transform(X_test_tensor.numpy())
        future_predictions_unscaled = scaler_Y.inverse_transform(future_predictions.numpy())
        future_dates_unscaled = scaler_X.inverse_transform(future_dates_scaled_tensor.numpy())

    for i in range(len(stock_data1) - 1):  # Loop until the second-to-last element
        stock1 = stock_data1.iloc[i]  # Get the row at index i
        stock2 = stock_data1.iloc[i + 1]  # Get the next row at index i+1

        # Calculate percentage change in 'Close' price
        pct_change = ((stock2['Close'] - stock1['Close']) / stock1['Close']) * 100

        # trying to determine why the stock changed by such an amount
        if pct_change > 3 or pct_change < -3:

            # get the dates
            if stock_data1.iloc[i-3] is not None:
                date_str = stock_data1.iloc[i-2]['Date']
            else:
                date_str = stock_data1.iloc[i]['Date']

            date_str = pd.to_datetime(date_str).strftime('%Y-%m-%d')

            # Finnhub API endpoint for company news
            url = f'https://finnhub.io/api/v1/company-news?symbol={stock}&from={date_str}&to={date_str}&token={API_KEY}'

             # Make a GET request to the API
            response = requests.get(url)

            print(f"{stock} changed by {pct_change:.2f}% on {date_str}")

            # Check if the request was successful
            if response.status_code == 200:
                news_data = response.json()
                
                # Print out the news articles
                if news_data:
                    print(f"News articles for {stock} on {date_str}:\n")
                    # Perform sentiment analysis on each article
                    sentiments = []
                    for article in news_data:
                        # Combine headline and summary to provide full context
                        text = article['headline'] + " " + article['summary']
                        if len(text) > 512:
                            text = text[:512]
                        sentiment = classifier(text)
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

                    total_articles = len(sentiments)
                    positive_percent = (positive / total_articles) * 100
                    negative_percent = (negative / total_articles) * 100
                    neutral_percent = (neutral / total_articles) * 100

                    print(f"Positive Sentiment: {positive_percent:.2f}%")
                    print(f"Negative Sentiment: {negative_percent:.2f}%")
                    print(f"Neutral Sentiment: {neutral_percent:.2f}%")

                    # Define thresholds and ranges for sentiment-based recommendations
                    strong_positive_threshold = 70
                    moderate_positive_threshold = 40
                    strong_negative_threshold = 70
                    moderate_negative_threshold = 40

                    # Market trend and price position checks
                    if (positive_percent >= strong_positive_threshold and stock1['Close'] <= (maxY - minY) * 0.2 + minY):
                        recommendation = "STRONG BUY"
                    elif (positive_percent >= moderate_positive_threshold and stock1['Close'] <= (maxY - minY) * 0.3 + minY):
                        recommendation = "BUY"
                    elif (positive_percent >= strong_positive_threshold):
                        recommendation = "BUY"
                    elif (negative_percent >= strong_negative_threshold and stock1['Close'] >= (maxY - minY) * 0.8 + minY):
                        recommendation = "STRONG SELL"
                    elif (negative_percent >= moderate_negative_threshold and stock1['Close'] >= (maxY - minY) * 0.7 + minY):
                        recommendation = "SELL"
                    elif (negative_percent >= strong_negative_threshold):
                        recommendation = "SELL"
                    else:
                        recommendation = "HOLD"

                    print(f"Recommendation: {recommendation}")
                    # Append the recommendation for each stock into the recommendations list
                    recommendations.append(recommendation)
                    
                else:
                    print(f"No news articles found for {stock} on {date_str}.")
            else:
                print(f"Error {response.status_code}: Unable to retrieve news.")
    
    # plot the predictions
    plotPrediction(X_train_original, Y_train_original, X_test_original, Y_test_original,
                    predictions=y_preds_original, symbol=stock, recommendations=recommendations,
                    future_preds=future_predictions_unscaled, future_dates_seconds=future_dates_seconds)
