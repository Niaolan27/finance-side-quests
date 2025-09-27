# Finance Side Quests

A comprehensive collection of Python scripts and tools for financial analysis, options pricing, data scraping, backtesting, and more. This repository provides modular, well-documented financial tools that can be used individually or combined for complex financial analysis workflows.

## 🚀 Features

- **Financial Data Scraping**: Fetch stock prices, financial statements, options data, and market indices
- **Options Pricing Models**: Black-Scholes, Binomial Trees, Monte Carlo simulations, and Greeks calculations
- **Technical Analysis**: 20+ technical indicators including RSI, MACD, Bollinger Bands, and more
- **Risk Metrics**: VaR, CVaR, Sharpe ratio, maximum drawdown, volatility analysis
- **Backtesting Framework**: Test trading strategies with realistic commission models
- **Financial Calculations**: Bond pricing, CAPM, present value, loan payments, and utility functions
- **Portfolio Analysis**: Performance metrics, risk analysis, and comparison tools

## 📁 Project Structure

```
finance-side-quests/
├── finance_tools/              # Core financial modules
│   ├── data_scraping/         # Data fetching utilities
│   │   └── scraper.py
│   ├── options_pricing/       # Options models and Greeks
│   │   └── models.py
│   ├── analysis/              # Technical indicators and risk metrics
│   │   └── indicators.py
│   ├── backtesting/          # Strategy backtesting framework
│   │   └── engine.py
│   └── utils/                # Common financial calculations
│       └── calculations.py
├── examples/                  # Example usage scripts
│   ├── stock_analysis_example.py
│   ├── options_pricing_example.py
│   └── backtesting_example.py
├── tests/                    # Unit tests (future)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Niaolan27/finance-side-quests.git
cd finance-side-quests
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Optional: Create a virtual environment**:
```bash
python -m venv finance_env
source finance_env/bin/activate  # On Windows: finance_env\Scripts\activate
pip install -r requirements.txt
```

## 📊 Quick Start Examples

### Data Scraping
```python
from finance_tools.data_scraping.scraper import DataScraper

scraper = DataScraper()

# Fetch stock data
aapl_data = scraper.get_stock_data("AAPL", period="1y")
print(aapl_data.head())

# Get financial statements
financials = scraper.get_financial_statements("AAPL")
print(financials['income_statement'].head())

# Fetch options data
options = scraper.get_options_data("AAPL")
print(options['calls'].head())
```

### Options Pricing
```python
from finance_tools.options_pricing.models import OptionsCalculator

calc = OptionsCalculator()

# Black-Scholes pricing
S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.2
call_price = calc.black_scholes(S, K, T, r, sigma, 'call')
print(f"Call Price: ${call_price:.2f}")

# Calculate Greeks
greeks = calc.calculate_greeks(S, K, T, r, sigma, 'call')
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")

# Implied volatility
market_price = 3.50
iv = calc.implied_volatility(market_price, S, K, T, r, 'call')
print(f"Implied Volatility: {iv:.1%}")
```

### Technical Analysis
```python
from finance_tools.analysis.indicators import TechnicalIndicators, RiskMetrics
from finance_tools.data_scraping.scraper import quick_stock_data

# Get data and calculate indicators
data = quick_stock_data("AAPL", "6mo")
tech = TechnicalIndicators()

# Technical indicators
rsi = tech.rsi(data['Close'])
macd = tech.macd(data['Close'])
bollinger = tech.bollinger_bands(data['Close'])

print(f"Current RSI: {rsi.iloc[-1]:.2f}")
print(f"MACD: {macd['macd'].iloc[-1]:.4f}")

# Risk metrics
risk = RiskMetrics()
returns = risk.calculate_returns(data['Close'])
vol = risk.volatility(returns)
sharpe = risk.sharpe_ratio(returns)

print(f"Annualized Volatility: {vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
```

### Backtesting
```python
from finance_tools.backtesting.engine import Backtester, MovingAverageCrossover
from finance_tools.data_scraping.scraper import quick_stock_data

# Get historical data
data = quick_stock_data("SPY", "2y")

# Create and test strategy
strategy = MovingAverageCrossover(fast_window=10, slow_window=30)
backtester = Backtester(strategy, initial_capital=100000)

results = backtester.run(data)
backtester.print_results()
```

## 🔧 Core Modules

### 1. Data Scraping (`finance_tools/data_scraping/`)
- **Stock Data**: Historical OHLCV data with customizable periods and intervals
- **Financial Statements**: Income statements, balance sheets, cash flow statements
- **Options Data**: Options chains with Greeks and implied volatility
- **Market Data**: Indices, sector ETFs, and market-wide metrics
- **Company Info**: Comprehensive company information and statistics

### 2. Options Pricing (`finance_tools/options_pricing/`)
- **Black-Scholes Model**: European options pricing with dividend support
- **Binomial Trees**: American and European options with customizable steps
- **Monte Carlo Simulation**: Path-dependent options and exotic derivatives
- **Greeks Calculator**: Delta, Gamma, Theta, Vega, Rho calculations
- **Implied Volatility**: Numerical solver for market-implied volatility
- **Portfolio Management**: Multi-option portfolio analysis and P&L tracking

### 3. Technical Analysis (`finance_tools/analysis/`)
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR
- **Risk Metrics**: VaR, CVaR, volatility, Sharpe ratio, Sortino ratio, maximum drawdown
- **Performance Analysis**: Returns analysis, win rates, risk-adjusted metrics
- **Rolling Analysis**: Time-varying metrics and rolling window calculations

### 4. Backtesting (`finance_tools/backtesting/`)
- **Strategy Framework**: Abstract base class for custom strategies
- **Built-in Strategies**: Moving average crossover, RSI, Bollinger Bands, momentum
- **Portfolio Management**: Realistic commission models, position sizing, cash management
- **Performance Metrics**: Comprehensive backtesting results with risk metrics
- **Equity Curve**: Track portfolio value over time with detailed trade history

### 5. Utilities (`finance_tools/utils/`)
- **Financial Calculations**: Present/future value, annuities, loan payments, bond pricing
- **Data Processing**: Cleaning, resampling, normalization, train/test splits
- **Portfolio Utils**: Rebalancing, equal weighting, market cap weighting
- **Trading Utils**: Position sizing, Kelly criterion, Fibonacci retracements

## 📈 Example Scripts

### Stock Analysis Example
```bash
python examples/stock_analysis_example.py
```
Demonstrates:
- Individual stock analysis with technical indicators
- Multi-stock comparison
- Sector analysis with ETFs
- Risk and performance metrics

### Options Pricing Example
```bash
python examples/options_pricing_example.py
```
Demonstrates:
- Black-Scholes vs Binomial vs Monte Carlo pricing
- Greeks analysis across different strikes
- Implied volatility calculations
- Multi-option portfolio strategies (Iron Condor)

### Backtesting Example
```bash
python examples/backtesting_example.py
```
Demonstrates:
- Single strategy backtesting
- Strategy comparison and ranking
- Parameter optimization
- Walk-forward analysis

## 🔍 Key Features in Detail

### Advanced Options Analysis
- **Multiple Pricing Models**: Compare results across Black-Scholes, Binomial, and Monte Carlo
- **Greeks Sensitivity**: Analyze how option prices change with market conditions  
- **Portfolio Greeks**: Aggregate Greeks for complex multi-option strategies
- **Implied Volatility**: Extract market expectations from option prices

### Comprehensive Risk Management
- **Value at Risk**: Quantify potential losses at various confidence levels
- **Maximum Drawdown**: Track peak-to-trough portfolio declines
- **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Beta Analysis**: Systematic risk relative to market benchmarks

### Professional Backtesting
- **Realistic Trading**: Commission costs, bid-ask spreads, slippage modeling
- **Multiple Strategies**: Built-in strategies plus framework for custom development  
- **Walk-Forward Analysis**: Test strategy robustness across different market regimes
- **Parameter Optimization**: Systematic approach to strategy tuning

### Real Market Data
- **Multiple Data Sources**: Yahoo Finance, Alpha Vantage integration
- **Live Options Data**: Real-time options chains and implied volatility
- **Financial Statements**: Fundamental analysis with quarterly/annual data
- **Market Indices**: Track broad market performance and sector rotation

## 🤝 Contributing

Contributions are welcome! Here are some areas where help would be appreciated:

1. **Additional Strategies**: More sophisticated trading strategies
2. **Alternative Data**: Integration with news, sentiment, or economic data
3. **Portfolio Optimization**: Modern Portfolio Theory, Black-Litterman
4. **Machine Learning**: ML-based trading strategies and feature engineering
5. **Testing**: Unit tests and integration tests
6. **Documentation**: More examples and tutorials

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Disclaimer

This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Trading and investing involve substantial risk of loss, and past performance does not guarantee future results. Always consult with qualified financial professionals before making investment decisions.

## 📧 Contact

For questions, suggestions, or collaborations:
- GitHub Issues: [Create an issue](https://github.com/Niaolan27/finance-side-quests/issues)
- Discussions: [GitHub Discussions](https://github.com/Niaolan27/finance-side-quests/discussions)

---

**Happy Trading! 📈**
