"""
Backtesting framework for testing trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.parameters = {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        
        Args:
            data: OHLCV data
        
        Returns:
            Series with trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        pass
    
    def set_parameters(self, **kwargs):
        """Set strategy parameters."""
        self.parameters.update(kwargs)


class MovingAverageCrossover(Strategy):
    """Simple moving average crossover strategy."""
    
    def __init__(self, fast_window: int = 10, slow_window: int = 20):
        """
        Initialize MA crossover strategy.
        
        Args:
            fast_window: Fast moving average window
            slow_window: Slow moving average window
        """
        super().__init__("MA Crossover")
        self.fast_window = fast_window
        self.slow_window = slow_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on moving average crossover."""
        prices = data['Close'] if 'Close' in data.columns else data['close']
        
        fast_ma = prices.rolling(window=self.fast_window).mean()
        slow_ma = prices.rolling(window=self.slow_window).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1  # Buy signal
        signals[fast_ma < slow_ma] = -1  # Sell signal
        
        # Generate signals only on crossovers
        signal_changes = signals.diff()
        final_signals = pd.Series(0, index=data.index)
        final_signals[signal_changes == 2] = 1   # Buy on crossover up
        final_signals[signal_changes == -2] = -1  # Sell on crossover down
        
        return final_signals


class RSIStrategy(Strategy):
    """RSI-based mean reversion strategy."""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Initialize RSI strategy.
        
        Args:
            rsi_period: RSI calculation period
            oversold: Oversold threshold
            overbought: Overbought threshold
        """
        super().__init__("RSI Strategy")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI levels."""
        prices = data['Close'] if 'Close' in data.columns else data['close']
        
        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1   # Buy when oversold
        signals[rsi > self.overbought] = -1  # Sell when overbought
        
        # Generate signals only on first breach
        buy_signals = (rsi < self.oversold) & (rsi.shift(1) >= self.oversold)
        sell_signals = (rsi > self.overbought) & (rsi.shift(1) <= self.overbought)
        
        final_signals = pd.Series(0, index=data.index)
        final_signals[buy_signals] = 1
        final_signals[sell_signals] = -1
        
        return final_signals


class BollingerBandStrategy(Strategy):
    """Bollinger Band mean reversion strategy."""
    
    def __init__(self, window: int = 20, num_std: float = 2):
        """
        Initialize Bollinger Band strategy.
        
        Args:
            window: Moving average window
            num_std: Number of standard deviations
        """
        super().__init__("Bollinger Band Strategy")
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on Bollinger Band touches."""
        prices = data['Close'] if 'Close' in data.columns else data['close']
        
        # Calculate Bollinger Bands
        middle_band = prices.rolling(window=self.window).mean()
        std = prices.rolling(window=self.window).std()
        upper_band = middle_band + (std * self.num_std)
        lower_band = middle_band - (std * self.num_std)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy when price touches lower band
        buy_signals = (prices <= lower_band) & (prices.shift(1) > lower_band.shift(1))
        # Sell when price touches upper band
        sell_signals = (prices >= upper_band) & (prices.shift(1) < upper_band.shift(1))
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals


class Portfolio:
    """Portfolio management for backtesting."""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = 0  # Number of shares held
        self.portfolio_value = []
        self.trades = []
        self.equity_curve = []
    
    def execute_trade(self, signal: int, price: float, timestamp: pd.Timestamp):
        """
        Execute a trade based on signal.
        
        Args:
            signal: Trading signal (1 for buy, -1 for sell)
            price: Execution price
            timestamp: Trade timestamp
        """
        if signal == 1 and self.cash > 0:  # Buy
            # Buy as many shares as possible
            shares_to_buy = int(self.cash / (price * (1 + self.commission)))
            if shares_to_buy > 0:
                cost = shares_to_buy * price * (1 + self.commission)
                self.cash -= cost
                self.positions += shares_to_buy
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': price,
                    'cost': cost
                })
        
        elif signal == -1 and self.positions > 0:  # Sell
            # Sell all positions
            proceeds = self.positions * price * (1 - self.commission)
            self.cash += proceeds
            
            self.trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'shares': self.positions,
                'price': price,
                'proceeds': proceeds
            })
            
            self.positions = 0
        
        # Update portfolio value
        current_value = self.cash + (self.positions * price)
        self.portfolio_value.append(current_value)
        self.equity_curve.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'positions': self.positions,
            'stock_value': self.positions * price,
            'total_value': current_value
        })


class Backtester:
    """Main backtesting engine."""
    
    def __init__(self, strategy: Strategy, initial_capital: float = 100000, 
                 commission: float = 0.001):
        """
        Initialize backtester.
        
        Args:
            strategy: Trading strategy to test
            initial_capital: Starting capital
            commission: Commission rate
        """
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital, commission)
        self.results = {}
    
    def run(self, data: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            data: Historical OHLCV data
        
        Returns:
            Dictionary with backtest results
        """
        # Reset portfolio
        self.portfolio.reset()
        
        # Generate signals
        signals = self.strategy.generate_signals(data)
        
        # Get prices for execution
        prices = data['Close'] if 'Close' in data.columns else data['close']
        
        # Execute trades
        for timestamp, signal in signals.items():
            if signal != 0:  # Only execute on non-zero signals
                price = prices.loc[timestamp]
                self.portfolio.execute_trade(signal, price, timestamp)
            else:
                # Still update portfolio value for equity curve
                current_price = prices.loc[timestamp]
                current_value = self.portfolio.cash + (self.portfolio.positions * current_price)
                self.portfolio.portfolio_value.append(current_value)
                self.portfolio.equity_curve.append({
                    'timestamp': timestamp,
                    'cash': self.portfolio.cash,
                    'positions': self.portfolio.positions,
                    'stock_value': self.portfolio.positions * current_price,
                    'total_value': current_value
                })
        
        # Calculate performance metrics
        self.results = self._calculate_performance(data, prices)
        return self.results
    
    def _calculate_performance(self, data: pd.DataFrame, prices: pd.Series) -> Dict:
        """Calculate performance metrics."""
        if not self.portfolio.equity_curve:
            return {}
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        portfolio_returns = equity_df['total_value'].pct_change().dropna()
        benchmark_returns = prices.pct_change().dropna()
        
        # Align returns
        aligned_returns = portfolio_returns.reindex(benchmark_returns.index).dropna()
        aligned_benchmark = benchmark_returns.reindex(aligned_returns.index)
        
        # Performance metrics
        total_return = (equity_df['total_value'].iloc[-1] / self.portfolio.initial_capital) - 1
        benchmark_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        
        # Annualized metrics
        trading_days = len(aligned_returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Risk metrics
        volatility = aligned_returns.std() * np.sqrt(252) if len(aligned_returns) > 1 else 0
        sharpe_ratio = (aligned_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = equity_df['total_value'].expanding().max()
        drawdown = (equity_df['total_value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = [t for t in self.portfolio.trades if t['action'] == 'SELL']
        if len(winning_trades) > 1:
            trade_returns = []
            buy_prices = {}
            
            for trade in self.portfolio.trades:
                if trade['action'] == 'BUY':
                    buy_prices[trade['timestamp']] = trade['price']
                elif trade['action'] == 'SELL':
                    # Find corresponding buy
                    for buy_time, buy_price in buy_prices.items():
                        if buy_time < trade['timestamp']:
                            trade_return = (trade['price'] - buy_price) / buy_price
                            trade_returns.append(trade_return)
                            del buy_prices[buy_time]
                            break
            
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0
            avg_win = np.mean([r for r in trade_returns if r > 0]) if trade_returns else 0
            avg_loss = np.mean([r for r in trade_returns if r < 0]) if trade_returns else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        return {
            'strategy_name': self.strategy.name,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(self.portfolio.trades),
            'final_portfolio_value': equity_df['total_value'].iloc[-1],
            'equity_curve': equity_df,
            'trades': self.portfolio.trades
        }
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if self.results and 'equity_curve' in self.results:
            return self.results['equity_curve']
        return pd.DataFrame()
    
    def print_results(self):
        """Print formatted backtest results."""
        if not self.results:
            print("No results available. Run backtest first.")
            return
        
        print(f"\n=== Backtest Results: {self.results['strategy_name']} ===")
        print(f"Total Return: {self.results['total_return']:.2%}")
        print(f"Annualized Return: {self.results['annualized_return']:.2%}")
        print(f"Benchmark Return: {self.results['benchmark_return']:.2%}")
        print(f"Excess Return: {self.results['excess_return']:.2%}")
        print(f"Volatility: {self.results['volatility']:.2%}")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {self.results['max_drawdown']:.2%}")
        print(f"Win Rate: {self.results['win_rate']:.2%}")
        print(f"Average Win: {self.results['avg_win']:.2%}")
        print(f"Average Loss: {self.results['avg_loss']:.2%}")
        print(f"Total Trades: {self.results['total_trades']}")
        print(f"Final Portfolio Value: ${self.results['final_portfolio_value']:,.2f}")


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate sample OHLCV data
    returns = np.random.normal(0.001, 0.02, 252)
    prices = 100 * (1 + returns).cumprod()
    
    # Add some trend and volatility
    trend = np.linspace(0, 0.2, 252)
    prices = prices * (1 + trend)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, 252)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 252))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 252))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 252)
    }, index=dates)
    
    # Test different strategies
    strategies = [
        MovingAverageCrossover(fast_window=10, slow_window=20),
        RSIStrategy(rsi_period=14, oversold=30, overbought=70),
        BollingerBandStrategy(window=20, num_std=2)
    ]
    
    print("Running backtests on sample data...\n")
    
    for strategy in strategies:
        backtester = Backtester(strategy, initial_capital=100000, commission=0.001)
        results = backtester.run(data)
        backtester.print_results()
        print("-" * 50)