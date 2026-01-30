import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class StockPredictor:
    """
    A simple stock price predictor using Linear Regression and 
    Technical Indicators (Moving Averages).
    """
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.model = LinearRegression()

    def fetch_data(self, period="2y"):
        """Downloads historical stock data using yfinance."""
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.download(self.ticker, period=period)
        
        if self.data.empty:
            raise ValueError("No data found for this ticker.")
            
        # Clean data: handle multi-index columns if they exist
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)

    def prepare_features(self):
        """Calculates indicators and prepares X (features) and y (target)."""
        df = self.data.copy()
        
        # Calculate Technical Indicators
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Target: The Close price of the NEXT day
        df['Target'] = df['Close'].shift(-1)
        
        # Remove rows with NaN values created by rolling/shifting
        df = df.dropna()
        
        # Features: Current Close, 10-day MA, 50-day MA
        self.X = df[['Close', 'MA10', 'MA50']]
        self.y = df['Target']
        
        return df

    def train_and_test(self):
        """Splits data and trains the Linear Regression model."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, shuffle=False
        )
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        # Evaluation
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"\nModel Results for {self.ticker}:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        return y_test, predictions

    def predict_tomorrow(self):
        """Predicts the price for the next trading day using the latest data."""
        latest_data = self.X.iloc[-1:].values
        prediction = self.model.predict(latest_data)
        return prediction[0]

def main():
    ticker_symbol = input("Enter Stock Ticker (e.g., AAPL, TSLA, BTC-USD): ").upper()
    
    predictor = StockPredictor(ticker_symbol)
    
    try:
        predictor.fetch_data()
        predictor.prepare_features()
        y_test, predictions = predictor.train_and_test()
        
        # Forecast tomorrow
        tomorrow_price = predictor.predict_tomorrow()
        print(f"\nPredicted Close Price for tomorrow: ${tomorrow_price:.2f}")
        
        # Plotting the results
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test.values, label="Actual Price", color='blue')
        plt.plot(y_test.index, predictions, label="Predicted Price", color='red', linestyle='--')
        plt.title(f"{ticker_symbol} Price Prediction (Test Set)")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
