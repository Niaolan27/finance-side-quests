"""
Utility functions for common financial calculations and data manipulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import math


class FinancialCalculations:
    """Class containing common financial calculation utilities."""
    
    @staticmethod
    def future_value(present_value: float, rate: float, periods: int, 
                    compounding_frequency: int = 1) -> float:
        """
        Calculate future value with compound interest.
        
        Args:
            present_value: Present value
            rate: Annual interest rate (as decimal)
            periods: Number of years
            compounding_frequency: Compounding frequency per year
        
        Returns:
            Future value
        """
        return present_value * (1 + rate / compounding_frequency) ** (compounding_frequency * periods)
    
    @staticmethod
    def present_value(future_value: float, rate: float, periods: int,
                     compounding_frequency: int = 1) -> float:
        """
        Calculate present value.
        
        Args:
            future_value: Future value
            rate: Annual interest rate (as decimal)
            periods: Number of years
            compounding_frequency: Compounding frequency per year
        
        Returns:
            Present value
        """
        return future_value / (1 + rate / compounding_frequency) ** (compounding_frequency * periods)
    
    @staticmethod
    def annuity_present_value(payment: float, rate: float, periods: int) -> float:
        """
        Calculate present value of annuity.
        
        Args:
            payment: Periodic payment
            rate: Periodic interest rate
            periods: Number of periods
        
        Returns:
            Present value of annuity
        """
        if rate == 0:
            return payment * periods
        return payment * (1 - (1 + rate) ** -periods) / rate
    
    @staticmethod
    def annuity_future_value(payment: float, rate: float, periods: int) -> float:
        """
        Calculate future value of annuity.
        
        Args:
            payment: Periodic payment
            rate: Periodic interest rate
            periods: Number of periods
        
        Returns:
            Future value of annuity
        """
        if rate == 0:
            return payment * periods
        return payment * ((1 + rate) ** periods - 1) / rate
    
    @staticmethod
    def loan_payment(principal: float, rate: float, periods: int) -> float:
        """
        Calculate loan payment amount.
        
        Args:
            principal: Loan principal
            rate: Periodic interest rate
            periods: Number of periods
        
        Returns:
            Periodic payment amount
        """
        if rate == 0:
            return principal / periods
        return principal * (rate * (1 + rate) ** periods) / ((1 + rate) ** periods - 1)
    
    @staticmethod
    def bond_price(face_value: float, coupon_rate: float, yield_rate: float,
                  periods: int, frequency: int = 2) -> float:
        """
        Calculate bond price.
        
        Args:
            face_value: Face value of bond
            coupon_rate: Annual coupon rate
            yield_rate: Required yield rate
            periods: Years to maturity
            frequency: Coupon payments per year
        
        Returns:
            Bond price
        """
        total_periods = periods * frequency
        periodic_coupon = (coupon_rate / frequency) * face_value
        periodic_yield = yield_rate / frequency
        
        # Present value of coupon payments
        if periodic_yield == 0:
            coupon_pv = periodic_coupon * total_periods
        else:
            coupon_pv = periodic_coupon * (1 - (1 + periodic_yield) ** -total_periods) / periodic_yield
        
        # Present value of face value
        face_value_pv = face_value / (1 + periodic_yield) ** total_periods
        
        return coupon_pv + face_value_pv
    
    @staticmethod
    def bond_duration(face_value: float, coupon_rate: float, yield_rate: float,
                     periods: int, frequency: int = 2) -> float:
        """
        Calculate Macaulay duration of a bond.
        
        Args:
            face_value: Face value of bond
            coupon_rate: Annual coupon rate
            yield_rate: Required yield rate
            periods: Years to maturity
            frequency: Coupon payments per year
        
        Returns:
            Macaulay duration in years
        """
        total_periods = periods * frequency
        periodic_coupon = (coupon_rate / frequency) * face_value
        periodic_yield = yield_rate / frequency
        
        bond_price = FinancialCalculations.bond_price(face_value, coupon_rate, 
                                                    yield_rate, periods, frequency)
        
        duration_sum = 0
        for t in range(1, total_periods + 1):
            if t == total_periods:
                cash_flow = periodic_coupon + face_value
            else:
                cash_flow = periodic_coupon
            
            pv_cash_flow = cash_flow / (1 + periodic_yield) ** t
            duration_sum += (t / frequency) * pv_cash_flow
        
        return duration_sum / bond_price
    
    @staticmethod
    def capm_expected_return(risk_free_rate: float, beta: float, 
                           market_return: float) -> float:
        """
        Calculate expected return using CAPM.
        
        Args:
            risk_free_rate: Risk-free rate
            beta: Beta coefficient
            market_return: Expected market return
        
        Returns:
            Expected return
        """
        return risk_free_rate + beta * (market_return - risk_free_rate)


class DataUtils:
    """Utility functions for data manipulation and processing."""
    
    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean financial data by handling missing values and outliers.
        
        Args:
            data: Raw financial data
        
        Returns:
            Cleaned data
        """
        cleaned_data = data.copy()
        
        # Forward fill missing values (common for financial data)
        cleaned_data = cleaned_data.fillna(method='ffill')
        
        # Remove remaining NaN values
        cleaned_data = cleaned_data.dropna()
        
        # Remove outliers (values beyond 3 standard deviations)
        for column in cleaned_data.select_dtypes(include=[np.number]).columns:
            mean = cleaned_data[column].mean()
            std = cleaned_data[column].std()
            cleaned_data = cleaned_data[
                (cleaned_data[column] >= mean - 3 * std) & 
                (cleaned_data[column] <= mean + 3 * std)
            ]
        
        return cleaned_data
    
    @staticmethod
    def resample_data(data: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Resample time series data to different frequency.
        
        Args:
            data: Time series data with datetime index
            frequency: Target frequency ('D', 'W', 'M', 'Q', 'Y')
        
        Returns:
            Resampled data
        """
        # Define aggregation rules for OHLCV data
        agg_rules = {}
        
        if 'Open' in data.columns or 'open' in data.columns:
            open_col = 'Open' if 'Open' in data.columns else 'open'
            agg_rules[open_col] = 'first'
        
        if 'High' in data.columns or 'high' in data.columns:
            high_col = 'High' if 'High' in data.columns else 'high'
            agg_rules[high_col] = 'max'
        
        if 'Low' in data.columns or 'low' in data.columns:
            low_col = 'Low' if 'Low' in data.columns else 'low'
            agg_rules[low_col] = 'min'
        
        if 'Close' in data.columns or 'close' in data.columns:
            close_col = 'Close' if 'Close' in data.columns else 'close'
            agg_rules[close_col] = 'last'
        
        if 'Volume' in data.columns or 'volume' in data.columns:
            volume_col = 'Volume' if 'Volume' in data.columns else 'volume'
            agg_rules[volume_col] = 'sum'
        
        # For other columns, use last value
        for col in data.columns:
            if col not in agg_rules:
                agg_rules[col] = 'last'
        
        return data.resample(frequency).agg(agg_rules).dropna()
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate logarithmic returns.
        
        Args:
            prices: Price series
        
        Returns:
            Log returns series
        """
        return np.log(prices / prices.shift(1)).dropna()
    
    @staticmethod
    def calculate_simple_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate simple returns.
        
        Args:
            prices: Price series
        
        Returns:
            Simple returns series
        """
        return prices.pct_change().dropna()
    
    @staticmethod
    def normalize_data(data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize numerical data.
        
        Args:
            data: Data to normalize
            method: Normalization method ('minmax', 'zscore')
        
        Returns:
            Normalized data
        """
        normalized_data = data.copy()
        
        for column in data.select_dtypes(include=[np.number]).columns:
            if method == 'minmax':
                min_val = data[column].min()
                max_val = data[column].max()
                normalized_data[column] = (data[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = data[column].mean()
                std_val = data[column].std()
                normalized_data[column] = (data[column] - mean_val) / std_val
        
        return normalized_data
    
    @staticmethod
    def split_train_test(data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split time series data into training and testing sets.
        
        Args:
            data: Time series data
            train_ratio: Ratio of data to use for training
        
        Returns:
            Tuple of (train_data, test_data)
        """
        split_index = int(len(data) * train_ratio)
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
        
        return train_data, test_data


class PortfolioUtils:
    """Utility functions for portfolio management."""
    
    @staticmethod
    def calculate_portfolio_returns(weights: np.array, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio returns given weights and asset returns.
        
        Args:
            weights: Portfolio weights
            returns: Asset returns DataFrame
        
        Returns:
            Portfolio returns series
        """
        return (returns * weights).sum(axis=1)
    
    @staticmethod
    def rebalance_portfolio(current_weights: np.array, target_weights: np.array,
                          prices: pd.Series, total_value: float,
                          transaction_cost: float = 0.001) -> Dict:
        """
        Calculate rebalancing trades needed.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            prices: Current asset prices
            total_value: Total portfolio value
            transaction_cost: Transaction cost rate
        
        Returns:
            Dictionary with rebalancing information
        """
        weight_diff = target_weights - current_weights
        trades = weight_diff * total_value
        
        # Calculate transaction costs
        total_trades = np.abs(trades).sum()
        transaction_costs = total_trades * transaction_cost
        
        return {
            'trades': trades,
            'transaction_costs': transaction_costs,
            'net_trades': trades - np.sign(trades) * transaction_costs / len(trades)
        }
    
    @staticmethod
    def equal_weight_portfolio(n_assets: int) -> np.array:
        """
        Generate equal weight portfolio.
        
        Args:
            n_assets: Number of assets
        
        Returns:
            Equal weight array
        """
        return np.ones(n_assets) / n_assets
    
    @staticmethod
    def market_cap_weights(market_caps: pd.Series) -> np.array:
        """
        Calculate market capitalization weights.
        
        Args:
            market_caps: Market capitalization values
        
        Returns:
            Market cap weights
        """
        return market_caps / market_caps.sum()


class TradingUtils:
    """Utility functions for trading operations."""
    
    @staticmethod
    def calculate_position_size(account_value: float, risk_percentage: float,
                              entry_price: float, stop_loss: float) -> int:
        """
        Calculate position size based on risk management.
        
        Args:
            account_value: Total account value
            risk_percentage: Risk percentage per trade
            entry_price: Entry price
            stop_loss: Stop loss price
        
        Returns:
            Position size (number of shares)
        """
        risk_amount = account_value * (risk_percentage / 100)
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        return int(risk_amount / risk_per_share)
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Args:
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
        
        Returns:
            Optimal bet fraction
        """
        if avg_loss == 0:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        return max(0, kelly_fraction)  # Don't allow negative fractions
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            high: High price
            low: Low price
        
        Returns:
            Dictionary with Fibonacci levels
        """
        diff = high - low
        
        return {
            '0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '78.6%': high - 0.786 * diff,
            '100%': low
        }


if __name__ == "__main__":
    # Example usage of utility functions
    
    # Financial calculations
    fc = FinancialCalculations()
    
    # Future value calculation
    fv = fc.future_value(1000, 0.05, 10)
    print(f"Future Value: ${fv:.2f}")
    
    # Bond price calculation
    bond_price = fc.bond_price(1000, 0.05, 0.06, 5)
    print(f"Bond Price: ${bond_price:.2f}")
    
    # CAPM expected return
    expected_return = fc.capm_expected_return(0.02, 1.2, 0.08)
    print(f"CAPM Expected Return: {expected_return:.2%}")
    
    # Data utilities with sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Calculate returns
    du = DataUtils()
    returns = du.calculate_simple_returns(sample_data['Close'])
    print(f"\nMean Daily Return: {returns.mean():.4f}")
    print(f"Daily Volatility: {returns.std():.4f}")
    
    # Trading utilities
    tu = TradingUtils()
    position_size = tu.calculate_position_size(10000, 2, 100, 95)
    print(f"\nPosition Size: {position_size} shares")
    
    # Kelly criterion
    kelly_fraction = tu.kelly_criterion(0.6, 0.05, 0.03)
    print(f"Kelly Fraction: {kelly_fraction:.2%}")
    
    # Fibonacci levels
    fib_levels = tu.fibonacci_retracement(120, 100)
    print(f"\nFibonacci Retracement Levels:")
    for level, price in fib_levels.items():
        print(f"{level}: ${price:.2f}")