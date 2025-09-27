"""
Financial data scraping module using various APIs and web sources.
Provides functions to fetch stock prices, financial statements, and market data.
"""

import yfinance as yf
import pandas as pd
import requests
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta


class DataScraper:
    """Main class for scraping financial data from various sources."""
    
    def __init__(self):
        """Initialize the data scraper."""
        self.session = requests.Session()
    
    def get_stock_data(self, symbol: str, period: str = "1y", 
                       interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock price data using yfinance.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data and additional columns
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols: List of stock ticker symbols
            period: Data period
        
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        data = {}
        for symbol in symbols:
            data[symbol] = self.get_stock_data(symbol, period)
        return data
    
    def get_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch financial statements for a given stock.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary containing income statement, balance sheet, and cash flow
        """
        try:
            ticker = yf.Ticker(symbol)
            return {
                'income_statement': ticker.financials,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cashflow
            }
        except Exception as e:
            print(f"Error fetching financial statements for {symbol}: {e}")
            return {}
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get comprehensive stock information including company details.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            print(f"Error fetching stock info for {symbol}: {e}")
            return {}
    
    def get_options_data(self, symbol: str, expiration_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch options data for a stock.
        
        Args:
            symbol: Stock ticker symbol
            expiration_date: Specific expiration date (YYYY-MM-DD), if None uses nearest expiration
        
        Returns:
            Dictionary containing calls and puts DataFrames
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            exp_dates = ticker.options
            if not exp_dates:
                return {}
            
            # Use specified date or nearest expiration
            if expiration_date and expiration_date in exp_dates:
                exp_date = expiration_date
            else:
                exp_date = exp_dates[0]  # Nearest expiration
            
            # Get options chain
            options = ticker.option_chain(exp_date)
            
            return {
                'calls': options.calls,
                'puts': options.puts,
                'expiration_date': exp_date
            }
        except Exception as e:
            print(f"Error fetching options data for {symbol}: {e}")
            return {}
    
    def get_market_indices(self, indices: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch major market indices data.
        
        Args:
            indices: List of index symbols, defaults to major US indices
        
        Returns:
            Dictionary with index data
        """
        if indices is None:
            indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P500, Dow, NASDAQ, Russell2000
        
        return self.get_multiple_stocks(indices)
    
    def get_sector_etfs(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for major sector ETFs.
        
        Returns:
            Dictionary with sector ETF data
        """
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financials': 'XLF',
            'Energy': 'XLE',
            'Consumer_Discretionary': 'XLY',
            'Industrials': 'XLI',
            'Consumer_Staples': 'XLP',
            'Utilities': 'XLU',
            'Materials': 'XLB',
            'Real_Estate': 'XLRE',
            'Communication': 'XLC'
        }
        
        data = {}
        for sector, etf in sector_etfs.items():
            data[sector] = self.get_stock_data(etf)
        
        return data


def quick_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Quick function to get stock data without initializing the class.
    
    Args:
        symbol: Stock ticker symbol
        period: Data period
    
    Returns:
        DataFrame with stock data
    """
    scraper = DataScraper()
    return scraper.get_stock_data(symbol, period)


if __name__ == "__main__":
    # Example usage
    scraper = DataScraper()
    
    # Get Apple stock data
    aapl_data = scraper.get_stock_data("AAPL", period="3mo")
    print(f"AAPL data shape: {aapl_data.shape}")
    print(aapl_data.head())
    
    # Get multiple stocks
    tech_stocks = scraper.get_multiple_stocks(["AAPL", "MSFT", "GOOGL"])
    print(f"\nFetched data for {len(tech_stocks)} tech stocks")
    
    # Get financial statements
    financials = scraper.get_financial_statements("AAPL")
    if financials:
        print(f"\nAvailable financial statements: {list(financials.keys())}")