import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
from transformers import pipeline
import requests
import time



data = pd.read_csv("sp500LastYear.csv")
data1 = pd.read_csv("sp500LastYear.csv") # only from last year for news purposes
unique_symbols = data['Symbol'].unique()

# Initialize scalers for X and Y
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Load the sentiment-analysis pipeline
classifier = pipeline('sentiment-analysis')

# Finnhub API key (replace with your own)
API_KEY = 'csnui6hr01qkfk595rn0csnui6hr01qkfk595rng'

plt.figure(figsize = (11, 7))
recommendations = [] # for the graph later on

# Store evaluations in a dictionary for each stock
evaluation_results = {stock: [] for stock in unique_symbols}

# Function to calculate the total number of accurate recommendations
def calculate_total_correct_recommendations(evaluation_results):
    total_correct = 0
    total_incorrect = 0

    for stock, evaluations in evaluation_results.items():
        for evaluation in evaluations:
            if evaluation['result'] == "Correct":
                total_correct += 1
            elif evaluation['result'] == "Incorrect":
                total_incorrect += 1

    print(f"Total Correct Recommendations: {total_correct}")
    print(f"Total Incorrect Recommendations: {total_incorrect}")
    return total_correct, total_incorrect

# Evaluate stock movement and store results
def evaluate_stock_movement_and_store(stock, recommendation, recommendation_date, stock_data):

    filtered_data = stock_data.loc[stock_data['Date'] == recommendation_date]
    if not filtered_data.empty:
        recommendation_price = filtered_data['Close'].values[0]
    else:
        print(f"No data found for recommendation_date: {recommendation_date}")
        recommendation_price = None  # or a default value

    # Ensure recommendation_date is converted to datetime
    if isinstance(recommendation_date, str):
        recommendation_date = datetime.strptime(recommendation_date, '%Y-%m-%d')  # Adjust the format if needed
    
    # Look 7 days ahead
    future_date = recommendation_date + timedelta(days=7)
    
    # Track price change after a set period (e.g., 7 days or 1 week)
    future_date = recommendation_date + timedelta(days=7)  # Look 7 days ahead
    future_stock_data = stock_data[stock_data['Date'] >= future_date]
    
    if future_stock_data.empty:
        print(f"No data after recommendation date for {stock}")
        return
    
    future_price = future_stock_data.iloc[0]['Close']  # Price after 7 days
    
    if future_price is None or recommendation_price is None:
        print("Invalid values detected. Skipping calculation.")
        return  # or handle the situation appropriately
    
    # Calculate price change
    price_change = ((future_price - recommendation_price) / recommendation_price) * 100  # Percentage change

    # Evaluate the recommendation
    if recommendation == "BUY" and price_change > 0:
        result = "Correct"
    elif recommendation == "STRONG BUY" and price_change > 0:
        result = "Correct"
    elif recommendation == "SELL" and price_change < 0:
            result = "Correct"
    elif recommendation == "STRONG SELL" and price_change < 0:
        result = "Correct"
    elif recommendation == "HOLD" and abs(price_change) < 5:  # Assuming a small movement is a 'hold' success
        result = "Correct"
    else:
        result = "Incorrect"
    
    # Append results to the evaluation dictionary
    evaluation_results[stock].append({
        'date': recommendation_date,
        'price': recommendation_price,
        'future_price': future_price,
        'recommendation': recommendation,
        'result': result
    })
    print(f"Stock: {stock}, Recommendation: {recommendation}, Price Change: {price_change:.2f}% -> {result}")


def plot_correct_evaluations(stock, stock_data, evaluations):
    # Ensure Date column is in datetime format
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Sort by date to avoid plotting issues
    stock_data = stock_data.sort_values(by='Date')

    # Filter correct evaluations
    evaluations = [e for e in evaluations]
    
    if not evaluations:
        print(f"No evaluations for {stock}")
        return

    # Plot stock price over time
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Date'], stock_data['Close'], label='Stock Price', color='blue')
    
    # Annotate correct evaluations
    for eval in evaluations:
        if eval['result'] == 'Incorrect':
            plt.scatter(eval['date'], eval['price'], color='red', label=f"{eval['result']} {eval['recommendation']}")
            plt.annotate(f"{eval['recommendation']}", 
                        (eval['date'], eval['price']), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center', 
                        fontsize=8)
        elif eval['result'] == 'Correct':
            plt.scatter(eval['date'], eval['price'], color='green', label=f"{eval['result']} {eval['recommendation']}")
            plt.annotate(f"{eval['recommendation']}", 
                        (eval['date'], eval['price']), 
                        textcoords="offset points",
                        xytext=(0, 10), 
                        ha='center', 
                        fontsize=8)

    # Adjust x-axis to show the full date range
    # Extract dates from evaluations (assuming it's a list of dictionaries)
    evaluation_dates = [eval['date'] for eval in evaluations]

    if evaluation_dates:  # Ensure the list is not empty
        plt.xlim(min(evaluation_dates), max(evaluation_dates))
    else:
        plt.xlim(stock_data['Date'].min(), stock_data['Date'].max())
    plt.title(f"Evaluations for {stock}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid()
    plt.show()


# plotting the predictions for the data
def plotPrediction(trainData,
                   trainLabels,
                   testData,
                   testLabels,
                   symbol,
                   predictions=None,
                   recommendations=None):

    # plot training data in blue
    plt.scatter(trainData, trainLabels, c="b", s=4, label = "Training Data")

    # plot test data in green
    plt.scatter(testData, testLabels, c='g', s=4, label = "Testing Data")

    if predictions is not None:
        # plot the predictions in red (predictions were made on the test data)
        plt.scatter(testData, predictions, c= 'r', s=4, label="predictions")

    # Add recommendation text at the corresponding points
    if recommendations is not None:

        for i in range(min(len(testData), len(predictions), len(recommendations))):
            # Adding the recommendation as text on the plot at the corresponding data point
            # Use annotate to add the recommendation with an arrow
            plt.annotate(
                recommendations[i]['recommendation'],  # Text to display
                xy=(testData[i], testLabels[i]),  # Coordinates of the point (x, y)
                xytext=(testData[i], testLabels[i] + 4),  # Position of the text (slightly above the point)
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1),  # Arrow properties
                fontsize=10,  # Text size
                color='black',  # Text color
                ha='center',  # Horizontal alignment
                va='bottom'  # Vertical alignment
            )

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
    stock_data = data[data['Symbol'] == stock]
    stock_data = stock_data.dropna(subset=['Close'])

    # data only from the last year
    stock_data1 = data1[data1['Symbol'] == stock]
    stock_data1 = stock_data1.dropna(subset=['Close'])
    
    # x-values are the date
    stock_data['Date'] = pd.to_datetime(data['Date'])
    timestamps = stock_data['Date'].astype(np.int64) // 10**9 # convert to seconds since epoch
    X = timestamps.to_numpy().reshape(-1, 1)  # Reshape for PyTorch

    # y-values are the values of the stocks
    Y = stock_data['Close'].to_numpy().reshape(-1, 1)  # Reshape for PyTorch

    # for news algorithm
    maxY = max(Y)
    minY = min(Y)

    # split into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

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

    model = StockModel(input_size=X_train.shape[1]) # use model created earlier
    loss = nn.MSELoss() # best loss function for stocks
    optimizer = optim.Adam(model.parameters(), lr=0.1) # best to start with Adam optimizer good for wide range of functions

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

    with torch.inference_mode():
        y_preds = model(X_test_tensor).numpy()
        y_preds_original = scaler_Y.inverse_transform(y_preds)
        Y_train_original = scaler_Y.inverse_transform(Y_train_tensor.numpy())
        Y_test_original = scaler_Y.inverse_transform(Y_test_tensor.numpy())
        X_train_original = scaler_X.inverse_transform(X_train_tensor.numpy())
        X_test_original = scaler_X.inverse_transform(X_test_tensor.numpy())
    

    for i in range(len(stock_data1) - 1):  # Loop until the second-to-last element
        stock1 = stock_data1.iloc[i]  # Get the row at index i
        stock2 = stock_data1.iloc[i + 1]  # Get the next row at index i+1

        # Calculate percentage change in 'Close' price
        pct_change = ((stock2['Close'] - stock1['Close']) / stock1['Close']) * 100

        # trying to determine why the stock changed by such an amount
        if pct_change > 3 or pct_change < -3:

            # get the dates
            if stock_data1.iloc[i-3] is not None:
                date_str = stock_data1.iloc[i-3]['Date']
            else:
                date_str = stock_data1.iloc[i]['Date']

            # Finnhub API endpoint for company news
            url = f'https://finnhub.io/api/v1/company-news?symbol={stock}&from={date_str}&to={date_str}&token={API_KEY}'

             # Make a GET request to the API
            response = requests.get(url)

            print(f"{stock} changed by {pct_change:.2f}% on {date_str}")

            # Check if the request was successful
            if response.status_code == 200:
                news_data = response.json()
                
                # Print out the news articles
                if news_data and len(recommendations) < 501:
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
                    recommendations.append({
                        'recommendation': recommendation,
                        'date': stock1['Date']})

                    time.sleep(2)
                    
                else:
                    break
                    print(f"No news articles found for {stock} on {date_str}.")
            else:
                print(f"Error {response.status_code}: Unable to retrieve news.")
        
    if len(recommendations) >= 500:
        break

    # Extract dates from recommendations (list of dictionaries)
    dates = [rec['date'] for rec in recommendations]  # Loop through recommendations
    values = [rec['recommendation'] for rec in recommendations]

    # Perform evaluations for this stock
    for i in range(len(recommendations)):
        # Pass the i-th recommendation and its associated date
        evaluate_stock_movement_and_store(stock, values[i], dates[i], stock_data)

    

# Calculate and display total correct recommendations
total_correct, total_incorrect = calculate_total_correct_recommendations(evaluation_results)
