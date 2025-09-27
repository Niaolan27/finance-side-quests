"""
Financial analysis tools including technical indicators, risk metrics, and performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """Class containing various technical analysis indicators."""
    
    @staticmethod
    def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: Price series
            window: Moving average window
        
        Returns:
            SMA series
        """
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def exponential_moving_average(prices: pd.Series, window: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Price series
            window: EMA window
        
        Returns:
            EMA series
        """
        return prices.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, 
                       num_std: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            window: Moving average window
            num_std: Number of standard deviations
        
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        return {
            'upper': middle_band + (std * num_std),
            'middle': middle_band,
            'lower': middle_band - (std * num_std)
        }
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            window: RSI window
        
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast: Fast EMA window
            slow: Slow EMA window
            signal: Signal line EMA window
        
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                            k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period
            d_period: %D period
        
        Returns:
            Dictionary with %K and %D lines
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: ATR window
        
        Returns:
            ATR series
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr


class RiskMetrics:
    """Class for calculating various risk metrics."""
    
    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price series
            method: 'simple' or 'log' returns
        
        Returns:
            Returns series
        """
        if method == 'simple':
            return prices.pct_change()
        elif method == 'log':
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError("Method must be 'simple' or 'log'")
    
    @staticmethod
    def volatility(returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            returns: Returns series
            annualize: Whether to annualize volatility
        
        Returns:
            Volatility
        """
        vol = returns.std()
        if annualize:
            # Assume 252 trading days per year
            vol *= np.sqrt(252)
        return vol
    
    @staticmethod
    def var(returns: pd.Series, confidence: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Returns series
            confidence: Confidence level (e.g., 0.05 for 95% VaR)
        
        Returns:
            VaR value
        """
        return returns.quantile(confidence)
    
    @staticmethod
    def cvar(returns: pd.Series, confidence: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Returns series
            confidence: Confidence level
        
        Returns:
            CVaR value
        """
        var_threshold = RiskMetrics.var(returns, confidence)
        return returns[returns <= var_threshold].mean()
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0) -> float:
        """
        Calculate Sharpe Ratio.
        
        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            Sharpe ratio
        """
        excess_returns = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
        return (excess_returns * 252) / RiskMetrics.volatility(returns)
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0) -> float:
        """
        Calculate Sortino Ratio.
        
        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            Sortino ratio
        """
        excess_returns = returns.mean() - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        if downside_deviation == 0:
            return float('inf') if excess_returns > 0 else 0
        
        return (excess_returns * 252) / downside_deviation
    
    @staticmethod
    def maximum_drawdown(prices: pd.Series) -> Dict[str, float]:
        """
        Calculate Maximum Drawdown.
        
        Args:
            prices: Price series
        
        Returns:
            Dictionary with max drawdown, peak, and trough values
        """
        # Calculate running maximum
        peak = prices.cummax()
        
        # Calculate drawdown
        drawdown = (prices - peak) / peak
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        peak_idx = peak[:max_dd_idx].idxmax()
        
        return {
            'max_drawdown': max_dd,
            'peak_date': peak_idx,
            'trough_date': max_dd_idx,
            'peak_value': peak.loc[peak_idx],
            'trough_value': prices.loc[max_dd_idx]
        }
    
    @staticmethod
    def beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate Beta relative to market.
        
        Args:
            stock_returns: Stock returns series
            market_returns: Market returns series
        
        Returns:
            Beta coefficient
        """
        # Align the series
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        stock_aligned = aligned_data.iloc[:, 0]
        market_aligned = aligned_data.iloc[:, 1]
        
        covariance = np.cov(stock_aligned, market_aligned)[0, 1]
        market_variance = np.var(market_aligned)
        
        return covariance / market_variance if market_variance != 0 else 0


class PerformanceAnalysis:
    """Class for comprehensive performance analysis."""
    
    @staticmethod
    def analyze_performance(prices: pd.Series, benchmark_prices: pd.Series = None,
                          risk_free_rate: float = 0) -> Dict[str, float]:
        """
        Comprehensive performance analysis.
        
        Args:
            prices: Price series
            benchmark_prices: Benchmark price series (optional)
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            Dictionary with performance metrics
        """
        returns = RiskMetrics.calculate_returns(prices).dropna()
        
        # Basic metrics
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(prices)) - 1
        volatility = RiskMetrics.volatility(returns)
        
        # Risk metrics
        sharpe = RiskMetrics.sharpe_ratio(returns, risk_free_rate)
        sortino = RiskMetrics.sortino_ratio(returns, risk_free_rate)
        max_dd = RiskMetrics.maximum_drawdown(prices)
        var_95 = RiskMetrics.var(returns, 0.05)
        cvar_95 = RiskMetrics.cvar(returns, 0.05)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd['max_drawdown'],
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'win_rate': (returns > 0).mean(),
            'avg_win': returns[returns > 0].mean(),
            'avg_loss': returns[returns < 0].mean()
        }
        
        # Add beta if benchmark is provided
        if benchmark_prices is not None:
            benchmark_returns = RiskMetrics.calculate_returns(benchmark_prices).dropna()
            metrics['beta'] = RiskMetrics.beta(returns, benchmark_returns)
            
            # Information Ratio
            excess_returns = returns - benchmark_returns
            aligned_excess = excess_returns.dropna()
            if len(aligned_excess) > 0 and aligned_excess.std() != 0:
                metrics['information_ratio'] = aligned_excess.mean() * 252 / (aligned_excess.std() * np.sqrt(252))
            else:
                metrics['information_ratio'] = 0
        
        return metrics
    
    @staticmethod
    def rolling_metrics(prices: pd.Series, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            prices: Price series
            window: Rolling window size
        
        Returns:
            DataFrame with rolling metrics
        """
        returns = RiskMetrics.calculate_returns(prices).dropna()
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        rolling_metrics['rolling_return'] = returns.rolling(window).sum()
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        rolling_metrics['rolling_sharpe'] = (returns.rolling(window).mean() * 252) / (returns.rolling(window).std() * np.sqrt(252))
        
        return rolling_metrics


if __name__ == "__main__":
    # Example usage with sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate sample price data
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
    prices = pd.Series(100 * (1 + returns).cumprod(), index=dates)
    
    # Technical Indicators
    tech = TechnicalIndicators()
    sma_20 = tech.simple_moving_average(prices, 20)
    rsi = tech.rsi(prices)
    bollinger = tech.bollinger_bands(prices)
    
    print("Technical Indicators Example:")
    print(f"Current Price: ${prices.iloc[-1]:.2f}")
    print(f"SMA(20): ${sma_20.iloc[-1]:.2f}")
    print(f"RSI: {rsi.iloc[-1]:.2f}")
    print(f"Bollinger Upper: ${bollinger['upper'].iloc[-1]:.2f}")
    
    # Risk Metrics
    risk = RiskMetrics()
    daily_returns = risk.calculate_returns(prices)
    vol = risk.volatility(daily_returns)
    sharpe = risk.sharpe_ratio(daily_returns)
    max_dd = risk.maximum_drawdown(prices)
    
    print(f"\nRisk Metrics:")
    print(f"Annualized Volatility: {vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Maximum Drawdown: {max_dd['max_drawdown']:.2%}")
    
    # Performance Analysis
    perf = PerformanceAnalysis()
    performance = perf.analyze_performance(prices)
    
    print(f"\nPerformance Analysis:")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Win Rate: {performance['win_rate']:.2%}")