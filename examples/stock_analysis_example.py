"""
Example script demonstrating how to fetch and analyze stock data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finance_tools.data_scraping.scraper import DataScraper
from finance_tools.analysis.indicators import TechnicalIndicators, RiskMetrics, PerformanceAnalysis
import matplotlib.pyplot as plt
import pandas as pd


def analyze_stock(symbol: str, period: str = "1y"):
    """
    Comprehensive stock analysis example.
    
    Args:
        symbol: Stock ticker symbol
        period: Analysis period
    """
    print(f"\n=== Analyzing {symbol} ===")
    
    # Initialize data scraper
    scraper = DataScraper()
    
    # Fetch stock data
    print("Fetching stock data...")
    data = scraper.get_stock_data(symbol, period)
    
    if data.empty:
        print(f"No data available for {symbol}")
        return
    
    print(f"Data fetched: {len(data)} trading days")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Basic information
    current_price = data['Close'].iloc[-1]
    print(f"\nCurrent Price: ${current_price:.2f}")
    
    # Technical Indicators
    tech = TechnicalIndicators()
    
    # Moving averages
    sma_20 = tech.simple_moving_average(data['Close'], 20)
    sma_50 = tech.simple_moving_average(data['Close'], 50)
    ema_12 = tech.exponential_moving_average(data['Close'], 12)
    
    print(f"\nTechnical Indicators:")
    print(f"SMA(20): ${sma_20.iloc[-1]:.2f}")
    print(f"SMA(50): ${sma_50.iloc[-1]:.2f}")
    print(f"EMA(12): ${ema_12.iloc[-1]:.2f}")
    
    # RSI
    rsi = tech.rsi(data['Close'])
    print(f"RSI(14): {rsi.iloc[-1]:.2f}")
    
    # Bollinger Bands
    bb = tech.bollinger_bands(data['Close'])
    print(f"Bollinger Bands:")
    print(f"  Upper: ${bb['upper'].iloc[-1]:.2f}")
    print(f"  Middle: ${bb['middle'].iloc[-1]:.2f}")
    print(f"  Lower: ${bb['lower'].iloc[-1]:.2f}")
    
    # MACD
    macd = tech.macd(data['Close'])
    print(f"MACD: {macd['macd'].iloc[-1]:.4f}")
    print(f"Signal: {macd['signal'].iloc[-1]:.4f}")
    
    # Risk Analysis
    risk = RiskMetrics()
    returns = risk.calculate_returns(data['Close'])
    
    volatility = risk.volatility(returns)
    sharpe = risk.sharpe_ratio(returns)
    var_95 = risk.var(returns)
    max_dd = risk.maximum_drawdown(data['Close'])
    
    print(f"\nRisk Metrics:")
    print(f"Annualized Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"95% VaR: {var_95:.2%}")
    print(f"Maximum Drawdown: {max_dd['max_drawdown']:.2%}")
    
    # Performance Analysis
    perf = PerformanceAnalysis()
    performance = perf.analyze_performance(data['Close'])
    
    print(f"\nPerformance Metrics:")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Win Rate: {performance['win_rate']:.2%}")
    print(f"Skewness: {performance['skewness']:.2f}")
    print(f"Kurtosis: {performance['kurtosis']:.2f}")
    
    return data, performance


def compare_stocks(symbols: list, period: str = "1y"):
    """
    Compare multiple stocks.
    
    Args:
        symbols: List of stock symbols
        period: Analysis period
    """
    print(f"\n=== Comparing Stocks: {', '.join(symbols)} ===")
    
    scraper = DataScraper()
    comparison_data = {}
    
    for symbol in symbols:
        data = scraper.get_stock_data(symbol, period)
        if not data.empty:
            perf = PerformanceAnalysis()
            performance = perf.analyze_performance(data['Close'])
            comparison_data[symbol] = performance
    
    if not comparison_data:
        print("No data available for comparison")
        return
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data).T
    
    print("\nComparison Results:")
    print("=" * 80)
    
    metrics_to_show = [
        'total_return', 'annualized_return', 'volatility', 
        'sharpe_ratio', 'max_drawdown', 'win_rate'
    ]
    
    for metric in metrics_to_show:
        if metric in comparison_df.columns:
            print(f"\n{metric.replace('_', ' ').title()}:")
            for symbol in symbols:
                if symbol in comparison_df.index and metric in comparison_df.columns:
                    value = comparison_df.loc[symbol, metric]
                    if 'return' in metric or 'drawdown' in metric or 'rate' in metric or 'volatility' in metric:
                        print(f"  {symbol}: {value:.2%}")
                    else:
                        print(f"  {symbol}: {value:.2f}")
    
    return comparison_df


def sector_analysis():
    """Analyze major sector ETFs."""
    print("\n=== Sector Analysis ===")
    
    scraper = DataScraper()
    sector_data = scraper.get_sector_etfs()
    
    if not sector_data:
        print("Unable to fetch sector data")
        return
    
    sector_performance = {}
    perf = PerformanceAnalysis()
    
    for sector, data in sector_data.items():
        if not data.empty:
            performance = perf.analyze_performance(data['Close'])
            sector_performance[sector] = performance
    
    # Create comparison
    sector_df = pd.DataFrame(sector_performance).T
    
    # Sort by total return
    if 'total_return' in sector_df.columns:
        sector_df = sector_df.sort_values('total_return', ascending=False)
        
        print("\nSector Performance (sorted by total return):")
        print("=" * 60)
        
        for sector in sector_df.index:
            total_ret = sector_df.loc[sector, 'total_return']
            volatility = sector_df.loc[sector, 'volatility']
            sharpe = sector_df.loc[sector, 'sharpe_ratio']
            
            print(f"{sector:20s}: {total_ret:6.2%} | Vol: {volatility:5.2%} | Sharpe: {sharpe:5.2f}")
    
    return sector_df


if __name__ == "__main__":
    # Example 1: Analyze individual stock
    try:
        stock_data, performance = analyze_stock("AAPL", period="1y")
        print("\n" + "="*50)
    except Exception as e:
        print(f"Error analyzing AAPL: {e}")
    
    # Example 2: Compare tech stocks
    try:
        tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        comparison = compare_stocks(tech_stocks, period="1y")
        print("\n" + "="*50)
    except Exception as e:
        print(f"Error comparing stocks: {e}")
    
    # Example 3: Sector analysis
    try:
        sector_analysis()
    except Exception as e:
        print(f"Error in sector analysis: {e}")
    
    print("\n=== Analysis Complete ===")