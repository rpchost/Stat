import pandas as pd
import numpy as np
import streamlit as st
import requests
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.models.formatters import DatetimeTickFormatter
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Function to fetch Bitcoin price data from CoinGecko API
def fetch_bitcoin_data(days=30):
    url = f'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}'
    response = requests.get(url)
    data = response.json()
    prices = data['prices']
    dates = [datetime.fromtimestamp(x[0] / 1000) for x in prices]
    prices = [x[1] for x in prices]
    bitcoin_data = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    bitcoin_data.set_index('Date', inplace=True)
    return bitcoin_data

# Fit LSTM model and generate forecasted prices
def forecast_prices_lstm(data, days_ahead):
    try:
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

        # Prepare data for LSTM
        train_data = scaled_data[:len(scaled_data) - days_ahead]
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        # Train the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

        # Prepare test data
        test_data = scaled_data[len(scaled_data) - 60 - days_ahead:]
        x_test = []
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Generate predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Create forecast series
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days_ahead)
        forecast_series = pd.Series(predictions.flatten(), index=forecast_index)
        return forecast_series, model
    except Exception as e:
        st.error(f"Error in LSTM model fitting: {e}")
        return pd.Series(dtype='float64'), None

# Fetch real Bitcoin price data
bitcoin_data = fetch_bitcoin_data(days=30)

# Check if bitcoin_data is empty or constant
if bitcoin_data.empty or bitcoin_data['Price'].nunique() <= 1:
    st.error("Bitcoin data is empty or constant. Please check the data source.")
else:
    # Create a slider for number of days to forecast
    days_ahead = st.slider('Days ahead to forecast', 1, 30, 5, key="forecast_days")

    # Fit LSTM model and generate forecasted prices
    forecast_series, fitted_model = forecast_prices_lstm(bitcoin_data['Price'], days_ahead)

    # Check if forecast_series is empty
    if forecast_series.empty:
        st.error("Forecasted data is empty. Please check the LSTM model and input data.")
    else:
        # Generate dataset with forecasted prices
        forecasted_data = pd.DataFrame({
            'Date': forecast_series.index,
            'Price': forecast_series.values
        })
        forecasted_data.set_index('Date', inplace=True)

        # Concatenate original data with forecasted data
        combined_data = pd.concat([bitcoin_data, forecasted_data])

        # Plot the data using Bokeh
        p = figure(x_axis_type="datetime", title="Bitcoin Price Forecast", height=350, width=800)
        p.line(bitcoin_data.index, bitcoin_data['Price'], color='blue', legend_label='Actual', line_width=2)
        p.line(forecasted_data.index, forecasted_data['Price'], color='red', legend_label='Forecast', line_width=2)

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Date", "@x{%F}"),
                ("Price", "@y{0,0.00}")
            ],
            formatters={
                '@x': 'datetime',  # use 'datetime' formatter for '@x' field
            },
            mode='vline'
        )
        p.add_tools(hover)

        # Format the x-axis to show all dates
        p.xaxis.formatter = DatetimeTickFormatter(days="%Y-%m-%d")

        # Display the plot in Streamlit
        st.bokeh_chart(p)

        # Move this section to display the chart first
        st.title("Bitcoin Price Forecast")

        st.write("Bitcoin Data:")
        st.write(bitcoin_data)

        st.write("Forecasted Prices:")
        st.write(forecast_series)

        st.write("Forecasted Data:")
        st.write(forecasted_data)
