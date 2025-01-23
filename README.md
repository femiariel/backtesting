# Backtesting trading strategies

This project implements a trading strategy based on moving averages (SMA and EMA) applied to stock price data. The main objective is to generate buy and sell signals and evaluate the performance of this strategy against a benchmark, the **S&P 500** index.

## Table of Contents
1. [Project Description](#project-description)
2. [Trading Strategy](#trading-strategy)
3. [Benchmarks Used](#benchmarks-used)
4. [Usage Instructions](#usage-instructions)

---

## Project Description
This project aims to:
- **Apply quantitative methods** to generate buy and sell signals based on technical indicators.
- **Backtest the strategy** on multiple stocks (e.g., AAPL, AMZN, AMD, etc.).
- **Compare the performance** of the strategy against the S&P 500, used as the benchmark.

The strategy uses technical indicators such as:
- **SMA** (*Simple Moving Average*)
- **EMA** (*Exponential Moving Average*)
- **RSI** (*Relative Strength Index*)

Additionally, candlestick patterns like **bearish engulfing** are integrated into the strategy to optimize entry and exit points.

---

## Trading Strategy

The strategy is based on the following criteria:

### Buy Criteria
- **Condition 1:** The price is above SMA120.
- **Condition 2:** SMA10 > EMA10 (positive crossover).
- **Condition 3:** Annual return exceeds 20%.
- **Condition 4:** RSI is between 30 and 70.

### Sell Criteria
- **Condition 1:** SMA10 < EMA10 (negative crossover).
- **Condition 2:** A bearish candlestick pattern ("bearish engulfing") is detected.
- **Condition 3:** Annual return falls below 20%.
- **Condition 4:** The price drops below SMA120.


## Benchmarks Used

The **S&P 500 (SPY)** index is used as a benchmark to compare the strategy’s performance. Portfolio performance is measured in terms of:

- **Annualized Return**
- **Maximum Drawdown (MDD)**

## Usage Instructions

1. **Initialize the project:**  
   Clone the repository and ensure all dependencies are installed.

2. **Run the code:**  
   Execute the main script to run the strategy, backtest it  and generate performance charts
   ```
   python backtesting.py
   
4. **Modify parameters:**  
   Change the stock tickers or SMA/EMA parameters directly in the code.
