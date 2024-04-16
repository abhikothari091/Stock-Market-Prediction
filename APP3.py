
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from newsapi import NewsApiClient  # Import News API library
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from bs4 import BeautifulSoup
import requests
import constants as ct  # You might need to import this file if it contains important constants or configurations
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from datetime import datetime, timedelta, date

# Set page configuration
st.set_page_config(
    page_title="Stock Market Prediction App",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

# Title
st.title('Stock Market Prediction :chart_with_upwards_trend: :chart_with_downwards_trend:')

bar = st.progress(0)

# Get user input for stock ticker
user_input = st.text_input("Enter Stock Ticker", 'ITC.NS')

# Define function to fetch Nifty 50 data
def fetch_nifty_data():
    nifty_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "HINDUNILVR.NS", "INFY.NS",  # Example tickers
                     "ICICIBANK.NS", "KOTAKBANK.NS", "ITC.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
                     "MARUTI.NS", "LT.NS", "SBI.NS", "HDFC.NS", "AXISBANK.NS",
                     "BAJAJFINSV.NS", "INDUSINDBK.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS",
                     "BANKNIFTY.NS", "POWERGRID.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "JSWSTEEL.NS",
                     "M&M.NS", "IOC.NS", "BPCL.NS", "COALINDIA.NS", "GRASIM.NS",
                     "SBIN.NS", "SHREECEM.NS", "DRREDDY.NS", "BRITANNIA.NS", "BAJFINANCE.NS",
                     "ADANIPORTS.NS", "DIVISLAB.NS", "TATACONSUM.NS", "CIPLA.NS", "UPL.NS",
                     "TECHM.NS", "HCLTECH.NS", "WIPRO.NS", "HDFCLIFE.NS", "TITAN.NS"]

    nifty_data = yf.download(nifty_tickers, period="1d")  # Fetch data for Nifty 50 stocks for 1 day
    return nifty_data

# Define function to calculate percentage change
def calculate_percentage_change(data):
    for ticker in data.columns.levels[1]:
        data["Percentage Change", ticker] = ((data["Close", ticker] - data["Open", ticker]) / data["Open", ticker]) * 100
    return data

# Fetch Nifty 50 stock data
nifty_data = fetch_nifty_data()

# Calculate percentage change for each stock
nifty_data = calculate_percentage_change(nifty_data)

# Display top 5 gainers and losers side by side
st.title("Top 5 Gainers and Losers of Nifty 50 Stocks")
st.subheader("Data for Today")

# Aggregate percentage changes for all stocks
percentage_changes = nifty_data["Percentage Change"].mean()

# Sort data based on percentage change
sorted_data = percentage_changes.sort_values(ascending=False)

# Split the screen into two columns
col1, col2 = st.columns(2)

# Display top 5 gainers in the first column
with col1:
    st.subheader("Top 5 Gainers:")
    top_gainers = sorted_data.head(5)
    st.write(top_gainers)

# Display top 5 losers in the second column
with col2:
    st.subheader("Top 5 Losers:")
    top_losers = sorted_data.tail(5)
    st.write(top_losers)

# Button to trigger further processing
if st.button("Process"):
    # Fetch data for the entered stock ticker
    itc = yf.Ticker(user_input)
    df = itc.history(period="max")

    end_date = date.today()
    start_date = end_date - timedelta(days=1470)
    tomorrow = end_date + timedelta(days=1)
    prices = itc.history(start='2010-01-01', end=end_date).Close

    # Describing the data
    st.subheader('Data from max time period :heavy_dollar_sign: ')
    with st.expander("Data Preview:"):
        st.dataframe(df.tail(20))

    # Plotting closing price vs time graph
    fig, fig1 = st.columns(2)
    with fig:
        st.subheader("Closing price :vs: Time Graph")
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.Close)
        st.pyplot(fig)

    # Calculating 100 and 200 days moving averages
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()

    with fig1:
        st.subheader("Closing Price :vs: 100 Days Moving Avg")
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(df.Close, 'b', label='Closing Price')
        plt.plot(ma100, 'r', label='100 Days Moving Average')
        plt.legend()
        st.pyplot(fig1)

    # Plotting closing price vs 200 days moving average
    fig2, fig3 = st.columns(2)
    with fig2:
        st.subheader("Closing Price :vs: 200 Days moving Average")
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(df.Close, 'r', label='Closing Price')
        plt.plot(ma200, 'b', label="200 Days Moving Average")
        plt.legend()
        st.pyplot(fig2)

    # Plotting closing price vs 100 days moving average vs 200 days moving average
    with fig3:
        st.subheader("Closing Price :vs: 100 Days MA :vs: 200 days MA")
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(df.Close, 'b', label='Closing Price')
        plt.plot(ma100, 'r', label="100 days moving Avg")
        plt.plot(ma200, 'y', label="200 days moving Avg")
        plt.legend()
        st.pyplot(fig1)

    bar.progress(25)
    # Scaling data for LSTM model
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
    test = pd.DataFrame(df['Close'][int(len(df) * 0.7):int(len(df))])

    # Load the LSTM model
    model = load_model("keras_BE_model_55.h5")

    # Testing part
    past_100_days = train.tail(100)
    final_df = pd.concat([past_100_days, train])
    ip_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, ip_data.shape[0]):
        x_test.append(ip_data[i - 100:i])
        y_test.append(ip_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_pred = model.predict(x_test)
    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]
    y_pred = y_pred * scale_factor
    y_test = y_test * scale_factor

    # Plotting Original vs Predicted Stock Price
    st.subheader("Original Stock Price :vs: Predicted Stock Price")
    fig4 = plt.figure(figsize=(14, 7))
    plt.plot(y_test, 'g', label='Original Price')
    plt.plot(y_pred, 'r', label='Predicted Price')
    plt.xlabel("Time-steps (Trading Sessions)")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig4)

    bar.progress(50)
    # Download additional data for LSTM prediction
    df = yf.download(user_input, period="150d")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data)

    # LSTM Predicted Value Chart
    if len(scaled_data) >= 100:
        sequences = []
        for i in range(100, len(scaled_data)):
            sequences.append(scaled_data[i - 100:i, 0])
        sequences = np.array(sequences)
        data_min = data.min()
        data_max = data.max()
        sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)

        # Make predictions only if there are enough data points
        y_pred2 = model.predict(sequences)
        y_pred2 = y_pred2.flatten()
        # Reverse scaling to get actual price predictions
        y_pred2 = y_pred2 * (data_max - data_min) + data_min

        # Calculate MAE and RMSE
        actual_price = scaled_data[-1, 0] * (data_max - data_min) + data_min

        # Calculate MAE and RMSE
        mae = mean_absolute_error([actual_price], [y_pred2[-1]])
        rmse = np.sqrt(mean_squared_error([actual_price], [y_pred2[-1]]))

        st.write("Predicted Stock Closing Price for Tomorrow:", y_pred2[-1])
        st.write("Mean Absolute Error (MAE):", mae)
        st.write("Root Mean Squared Error (RMSE):", rmse)

    else:
        st.exception("Not enough data points for prediction.")

    # ARIMA model prediction
    model5 = sm.tsa.statespace.SARIMAX(prices, order=(2, 1, 2), seasonal_order=(2, 1, 2, 12))
    model5 = model5.fit()
    # Forecasting future values
    forecast1 = model5.predict(n_periods=len(prices))

    # Plotting ARIMA Predictions
    st.subheader("SARIMA Predictions vs Actual Prices")
    plt.figure(figsize=(14, 7))
    plt.plot(prices, label='Actual', color='blue')
    plt.plot(forecast1, label='Predictions', color='orange')

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('SARIMA Predictions vs Actual Prices')
    plt.legend()
    plt.show()
    st.pyplot(plt)
    bar.progress(75)
    # Forecasting future values for the next 30 days
    forecast_next_30_days = model5.forecast(steps=30)

    # Generate x-axis index for the forecasted values
    forecast_index = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

    # Filter prices data for dates after 2020
    prices_after_2020 = prices[prices.index.year >= 2023]

    # Plotting ARIMA Predictions for the next 30 days along with actual prices after 2020
    st.subheader("ARIMA PREDICTIONS")
    plt.figure(figsize=(14, 7))
    plt.plot(prices_after_2020.index, prices_after_2020.values, label='Actual (After 2020)')
    plt.plot(forecast_index, forecast_next_30_days, label='Predictions')

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('SARIMA Predictions for Next 30 Days vs Actual Prices (After 2020)')
    plt.legend()
    plt.show()
    st.pyplot(plt)

    # Forecasting next day's price
    next_day_forecast = model5.forecast(steps=1)
    next_day_price = next_day_forecast.iloc[0]

    # Display the predicted price for the next day
    st.write("Predicted ARIMA Price for Tomorrow:", next_day_price)

    # Sentiment Analysis
    # Function to retrieve news sentiment using News API
    def retrieve_news_sentiment(api_key, query, language='en', num_articles=5):
        newsapi = NewsApiClient(api_key=api_key)

        # Fetch news articles based on the query
        news_articles = newsapi.get_everything(q=query, language=language, sort_by='relevancy', page_size=num_articles)

        headlines = [article['title'] for article in news_articles['articles']]

        # Create a DataFrame for news headlines
        df_news = pd.DataFrame({'Headlines': headlines})

        # Perform sentiment analysis on news
        df_news['Sentiment'] = df_news['Headlines'].apply(lambda x: TextBlob(x).sentiment.polarity)

        # Categorize sentiment
        df_news['Sentiment_Category'] = np.where(df_news['Sentiment'] > 0, 'Positive',
                                                 np.where(df_news['Sentiment'] < 0, 'Negative', 'Neutral'))

        return df_news

    # Use the News API function in script
    news_api_key = 'df514b8ad7f148bbaeb0849561f617e1'  
    news_query = 'Indian stock market'  # Modify the query based on your needs
    df_news = retrieve_news_sentiment(news_api_key, news_query)

    # Display sentiment analysis for news
    st.subheader("Sentiment Analysis for Golbal News Headlines")
    st.write(df_news[['Headlines', 'Sentiment_Category']])

    # Recommendation Section
    mean = df.Close.tail(1).mean()

    # Define a function for recommending based on sentiment and stock data
    def recommending_p(df, lstm_predicted_price):
        # Initialize idea and decision variables
        idea = ""
        decision = ""

        if 'Close' in df.columns:
            last_close_price = df['Close'].iloc[-1]

            # Check if the LSTM predicted price is higher, lower, or equal to the last close price
            if lstm_predicted_price > last_close_price:
                idea = "RISE"
                decision = "BUY"
            elif lstm_predicted_price < last_close_price:
                idea = "FALL"
                decision = "SELL"
            else:
                idea = "HOLD"
                decision = "HOLD"

            # Define colors based on recommendations
            idea_color = "green" if idea == "RISE" else "red" if idea == "FALL" else "grey"
            decision_color = "green" if decision == "BUY" else "red" if decision == "SELL" else "grey"

            # Highlight the recommendations and make them stand out
            st.markdown(f'<p style="color:{idea_color}; font-weight:bold; font-size:16px;">Idea: {idea}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:{decision_color}; font-weight:bold; font-size:16px;">Decision: {decision}</p>', unsafe_allow_html=True)

        else:
            st.write("DataFrame does not contain 'Close' column. Columns available:", df.columns)

        return idea, decision

    def recommending_arima(df, arima_predicted_price):
        # Initialize idea and decision variables
        idea = ""
        decision = ""

        if 'Close' in df.columns:
            last_close_price = df['Close'].iloc[-1]

            # Check if the ARIMA predicted price is higher, lower, or equal to the last close price
            if arima_predicted_price > last_close_price:
                idea = "RISE"
                decision = "BUY"
            elif arima_predicted_price < last_close_price:
                idea = "FALL"
                decision = "SELL"
            else:
                idea = "HOLD"
                decision = "HOLD"

            # Define colors based on recommendations
            idea_color = "green" if idea == "RISE" else "red" if idea == "FALL" else "grey"
            decision_color = "green" if decision == "BUY" else "red" if decision == "SELL" else "grey"

            # Highlight the recommendations and make them stand out
            st.markdown(f'<p style="color:{idea_color}; font-weight:bold; font-size:16px;">Idea: {idea}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:{decision_color}; font-weight:bold; font-size:16px;">Decision: {decision}</p>', unsafe_allow_html=True)

        else:
            st.write("DataFrame does not contain 'Close' column. Columns available:", df.columns)

        return idea, decision


    def recommending(df, mean, df_news):
        # Initialize idea and decision variables
        idea = ""
        decision = ""

        if 'Close' in df.columns:
            last_close_price = df['Close'].iloc[-1]

            # Calculate sentiment counts
            positive_count = len(df_news[df_news['Sentiment_Category'] == 'Positive'])
            negative_count = len(df_news[df_news['Sentiment_Category'] == 'Negative'])
            neutral_count = len(df_news[df_news['Sentiment_Category'] == 'Neutral'])

            # Make recommendation based on sentiment counts
            if positive_count > negative_count and positive_count > neutral_count:
                idea = "RISE"
                decision = "BUY"
            elif negative_count > positive_count and negative_count > neutral_count:
                idea = "FALL"
                decision = "SELL"
            else:
                idea = "HOLD"
                decision = "HOLD"

            # Define colors based on sentiment counts
            idea_color = "green" if idea == "RISE" else "red" if idea == "FALL" else "grey"
            decision_color = "green" if decision == "BUY" else "red" if decision == "SELL" else "grey"

            # Highlight the sentiment counts and recommendations
            st.write(f"Number of Positive Sentiments: {positive_count}")
            st.write(f"Number of Negative Sentiments: {negative_count}")
            st.write(f"Number of Neutral Sentiments: {neutral_count}")
            st.markdown(f'<p style="color:{idea_color}; font-weight:bold; font-size:16px;">Idea: {idea}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:{decision_color}; font-weight:bold; font-size:16px;">Decision: {decision}</p>', unsafe_allow_html=True)

        else:
            st.write("DataFrame does not contain 'Close' column. Columns available:", df.columns)

        return idea, decision


    # Scrape news headlines
    news_url = 'https://economictimes.indiatimes.com/markets/stocks?from=mdr'
    response = requests.get(news_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [headline.text.strip() for headline in soup.find_all('h2')]

    # Create a DataFrame for news headlines
    df_news = pd.DataFrame({'Headlines': headlines})

    # Perform sentiment analysis on news
    df_news['Sentiment'] = df_news['Headlines'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Categorize sentiment
    df_news['Sentiment_Category'] = np.where(df_news['Sentiment'] > 0, 'Positive',
                                             np.where(df_news['Sentiment'] < 0, 'Negative', 'Neutral'))

    # Display sentiment analysis for news
    st.subheader("Sentiment Analysis for News Headlines on Economic Times")
    st.write(df_news[['Headlines', 'Sentiment_Category']])

    st.subheader("Recomendation based on Sentiments Analysis")
    # Recommendation Section
    recommending(df, mean, df_news)

    st.subheader("Recommendation based on the Predicted Price of LSTM")
    recommending_p(df, y_pred2[-1])
    #st.dataframe(df.tail(10))

    st.subheader("Recommendation based on the Predicted Price of ARIMA")
    recommending_p(df, next_day_price)

    bar.progress(100)
    st.success("Success")
    st.toast("Page Loaded")
    # Disclaimer Section
    st.markdown(
        '<hr>'  # Adds a horizontal line to separate sections
        '<h3>Disclaimer</h3>'
        '<p style="color:red; font-weight:bold;">'
        'This is a long-term advisory model. Investing in the securities market involves inherent risks. '
        'Make informed decisions after conducting thorough research and analysis. '
        'The predictions provided are based on historical data and machine learning models, which may not guarantee future outcomes. '
        'The accuracy of predictions may vary, and past performance is not indicative of future results. '
        'Always consult with a financial advisor and invest at your own risk. '
        'The developers and the application are not responsible for any financial losses incurred as a result of using this information.'
        '</p>', unsafe_allow_html=True
    )
