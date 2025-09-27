"""
Example script demonstrating backtesting strategies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finance_tools.backtesting.engine import (
    Backtester, MovingAverageCrossover, RSIStrategy, 
    BollingerBandStrategy, Strategy
)
from finance_tools.data_scraping.scraper import DataScraper
from finance_tools.analysis.indicators import TechnicalIndicators
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MomentumStrategy(Strategy):
    """Custom momentum strategy based on rate of change."""
    
    def __init__(self, lookback_period: int = 20, threshold: float = 0.05):
        super().__init__("Momentum Strategy")
        self.lookback_period = lookback_period
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on price momentum."""
        prices = data['Close'] if 'Close' in data.columns else data['close']
        
        # Calculate rate of change
        roc = (prices - prices.shift(self.lookback_period)) / prices.shift(self.lookback_period)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy when momentum is strong positive
        buy_signals = (roc > self.threshold) & (roc.shift(1) <= self.threshold)
        # Sell when momentum turns negative
        sell_signals = (roc < -self.threshold/2) & (roc.shift(1) >= -self.threshold/2)
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals


def backtest_single_strategy():
    """Backtest a single strategy with real data."""
    print("=== Single Strategy Backtest ===")
    
    try:
        # Fetch data
        scraper = DataScraper()
        symbol = "AAPL"
        data = scraper.get_stock_data(symbol, period="2y")
        
        if data.empty:
            print("No data available, using synthetic data instead")
            return create_synthetic_backtest()
        
        print(f"Backtesting {symbol} with {len(data)} days of data")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Initialize strategy
        strategy = MovingAverageCrossover(fast_window=10, slow_window=30)
        
        # Run backtest
        backtester = Backtester(strategy, initial_capital=100000, commission=0.001)
        results = backtester.run(data)
        
        # Print results
        backtester.print_results()
        
        # Get equity curve for plotting
        equity_curve = backtester.get_equity_curve()
        
        if not equity_curve.empty:
            print(f"\nEquity Curve Summary:")
            print(f"Initial Value: ${equity_curve['total_value'].iloc[0]:,.2f}")
            print(f"Final Value: ${equity_curve['total_value'].iloc[-1]:,.2f}")
            print(f"Peak Value: ${equity_curve['total_value'].max():,.2f}")
            print(f"Trough Value: ${equity_curve['total_value'].min():,.2f}")
        
        return results, data
        
    except Exception as e:
        print(f"Error in real data backtest: {e}")
        return create_synthetic_backtest()


def create_synthetic_backtest():
    """Create a backtest with synthetic data."""
    print("\n=== Synthetic Data Backtest ===")
    
    # Generate synthetic OHLCV data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    
    # Generate price series with trend and volatility
    returns = np.random.normal(0.0005, 0.02, 500)  # Small positive drift
    prices = 100 * (1 + returns).cumprod()
    
    # Add some cyclical patterns
    cycle = 0.05 * np.sin(np.linspace(0, 4 * np.pi, 500))
    prices = prices * (1 + cycle)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.002, 500)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 500))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 500))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)
    
    print(f"Generated synthetic data: {len(data)} days")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Test strategy
    strategy = MovingAverageCrossover(fast_window=10, slow_window=30)
    backtester = Backtester(strategy, initial_capital=100000, commission=0.001)
    results = backtester.run(data)
    
    backtester.print_results()
    
    return results, data


def compare_strategies():
    """Compare multiple strategies on the same data."""
    print("\n=== Strategy Comparison ===")
    
    try:
        # Try to get real data first
        scraper = DataScraper()
        data = scraper.get_stock_data("SPY", period="2y")  # Use SPY for broad market
        
        if data.empty:
            print("Using synthetic data for comparison")
            np.random.seed(42)
            dates = pd.date_range('2022-01-01', periods=500, freq='D')
            returns = np.random.normal(0.0005, 0.015, 500)
            prices = 100 * (1 + returns).cumprod()
            
            data = pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.002, 500)),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.008, 500))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.008, 500))),
                'Close': prices,
                'Volume': np.random.randint(500000, 5000000, 500)
            }, index=dates)
        
        # Define strategies to compare
        strategies = [
            MovingAverageCrossover(fast_window=5, slow_window=15),
            MovingAverageCrossover(fast_window=10, slow_window=30),
            RSIStrategy(rsi_period=14, oversold=30, overbought=70),
            BollingerBandStrategy(window=20, num_std=2),
            MomentumStrategy(lookback_period=20, threshold=0.05)
        ]
        
        results_comparison = {}
        
        print(f"Comparing {len(strategies)} strategies on {len(data)} days of data")
        print("-" * 80)
        
        for strategy in strategies:
            backtester = Backtester(strategy, initial_capital=100000, commission=0.001)
            results = backtester.run(data)
            results_comparison[strategy.name] = results
        
        # Create comparison table
        comparison_metrics = [
            'total_return', 'annualized_return', 'volatility', 
            'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades'
        ]
        
        print(f"\n{'Strategy':<25} {'Total Ret':<10} {'Ann Ret':<10} {'Sharpe':<8} {'Max DD':<8} {'Trades':<8}")
        print("-" * 75)
        
        for strategy_name, results in results_comparison.items():
            if results:  # Check if results are not empty
                total_ret = results.get('total_return', 0)
                ann_ret = results.get('annualized_return', 0) 
                sharpe = results.get('sharpe_ratio', 0)
                max_dd = results.get('max_drawdown', 0)
                trades = results.get('total_trades', 0)
                
                print(f"{strategy_name:<25} {total_ret:<10.1%} {ann_ret:<10.1%} "
                      f"{sharpe:<8.2f} {max_dd:<8.1%} {trades:<8d}")
        
        return results_comparison, data
        
    except Exception as e:
        print(f"Error in strategy comparison: {e}")
        return {}, pd.DataFrame()


def parameter_optimization():
    """Demonstrate parameter optimization for a strategy."""
    print("\n=== Parameter Optimization ===")
    
    # Generate test data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=300, freq='D')
    returns = np.random.normal(0.001, 0.018, 300)
    
    # Add some trend
    trend = np.linspace(0, 0.3, 300)
    returns = returns + trend / 300
    
    prices = 100 * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.003, 300)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 300))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 300))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 8000000, 300)
    }, index=dates)
    
    print("Optimizing Moving Average Crossover parameters...")
    
    # Test different parameter combinations
    fast_windows = [5, 10, 15, 20]
    slow_windows = [20, 30, 40, 50]
    
    best_sharpe = -999
    best_params = {}
    optimization_results = []
    
    for fast in fast_windows:
        for slow in slow_windows:
            if fast >= slow:  # Skip invalid combinations
                continue
                
            strategy = MovingAverageCrossover(fast_window=fast, slow_window=slow)
            backtester = Backtester(strategy, initial_capital=100000, commission=0.001)
            results = backtester.run(data)
            
            if results and 'sharpe_ratio' in results:
                sharpe = results['sharpe_ratio']
                total_return = results['total_return']
                max_dd = results['max_drawdown']
                trades = results['total_trades']
                
                optimization_results.append({
                    'fast_window': fast,
                    'slow_window': slow,
                    'sharpe_ratio': sharpe,
                    'total_return': total_return,
                    'max_drawdown': max_dd,
                    'total_trades': trades
                })
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'fast_window': fast, 'slow_window': slow}
    
    # Display optimization results
    print(f"\n{'Fast':<6} {'Slow':<6} {'Sharpe':<8} {'Total Ret':<10} {'Max DD':<8} {'Trades':<8}")
    print("-" * 52)
    
    for result in sorted(optimization_results, key=lambda x: x['sharpe_ratio'], reverse=True)[:10]:
        print(f"{result['fast_window']:<6d} {result['slow_window']:<6d} "
              f"{result['sharpe_ratio']:<8.2f} {result['total_return']:<10.1%} "
              f"{result['max_drawdown']:<8.1%} {result['total_trades']:<8d}")
    
    print(f"\nBest Parameters: Fast={best_params['fast_window']}, Slow={best_params['slow_window']}")
    print(f"Best Sharpe Ratio: {best_sharpe:.2f}")
    
    return optimization_results


def walk_forward_analysis():
    """Demonstrate walk-forward analysis."""
    print("\n=== Walk-Forward Analysis ===")
    
    # Generate longer time series for walk-forward
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=800, freq='D')
    
    # Create regime changes in the data
    returns = np.concatenate([
        np.random.normal(0.001, 0.015, 200),   # Bull market
        np.random.normal(-0.0005, 0.025, 200),  # Bear market  
        np.random.normal(0.0008, 0.012, 200),   # Recovery
        np.random.normal(0.0003, 0.018, 200)    # Sideways
    ])
    
    prices = 100 * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.003, 800)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 800))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 800))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 8000000, 800)
    }, index=dates)
    
    # Walk-forward parameters
    train_window = 200  # Training window size
    test_window = 50    # Test window size
    step_size = 25      # Step size for walk-forward
    
    walk_forward_results = []
    
    print(f"Running walk-forward analysis:")
    print(f"Training window: {train_window} days")
    print(f"Test window: {test_window} days")
    print(f"Step size: {step_size} days")
    
    for i in range(0, len(data) - train_window - test_window, step_size):
        # Define train and test periods
        train_start = i
        train_end = i + train_window
        test_start = train_end
        test_end = test_start + test_window
        
        if test_end > len(data):
            break
        
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]
        
        # In a real implementation, you would optimize parameters on train_data
        # For simplicity, we'll use fixed parameters
        strategy = MovingAverageCrossover(fast_window=10, slow_window=20)
        
        # Test on out-of-sample data
        backtester = Backtester(strategy, initial_capital=100000, commission=0.001)
        results = backtester.run(test_data)
        
        if results:
            walk_forward_results.append({
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'return': results['total_return'],
                'sharpe': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown']
            })
    
    if walk_forward_results:
        # Analyze walk-forward results
        avg_return = np.mean([r['return'] for r in walk_forward_results])
        avg_sharpe = np.mean([r['sharpe'] for r in walk_forward_results])
        worst_dd = min([r['max_drawdown'] for r in walk_forward_results])
        
        print(f"\nWalk-Forward Results ({len(walk_forward_results)} periods):")
        print(f"Average Return per Period: {avg_return:.1%}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Worst Max Drawdown: {worst_dd:.1%}")
        
        # Show period-by-period results
        print(f"\n{'Period':<20} {'Return':<10} {'Sharpe':<8} {'Max DD':<8}")
        print("-" * 48)
        
        for result in walk_forward_results[-5:]:  # Show last 5 periods
            period = f"{result['test_start'].strftime('%Y-%m')} - {result['test_end'].strftime('%Y-%m')}"
            print(f"{period:<20} {result['return']:<10.1%} {result['sharpe']:<8.2f} {result['max_drawdown']:<8.1%}")
    
    return walk_forward_results


if __name__ == "__main__":
    print("Starting comprehensive backtesting examples...\n")
    
    try:
        # Example 1: Single strategy backtest
        results, data = backtest_single_strategy()
        print("\n" + "="*60)
        
        # Example 2: Strategy comparison
        comparison, comp_data = compare_strategies()
        print("\n" + "="*60)
        
        # Example 3: Parameter optimization
        optimization = parameter_optimization()
        print("\n" + "="*60)
        
        # Example 4: Walk-forward analysis
        walk_forward = walk_forward_analysis()
        
    except Exception as e:
        print(f"Error running backtesting examples: {e}")
    
    print("\n=== Backtesting Examples Complete ===")