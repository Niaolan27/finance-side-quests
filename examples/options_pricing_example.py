"""
Example script demonstrating options pricing and Greeks calculations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finance_tools.options_pricing.models import OptionsCalculator, OptionPortfolio
from finance_tools.data_scraping.scraper import DataScraper
import numpy as np
import pandas as pd


def basic_options_pricing():
    """Demonstrate basic options pricing models."""
    print("=== Basic Options Pricing ===")
    
    # Option parameters
    S = 100      # Current stock price
    K = 105      # Strike price
    T = 0.25     # 3 months to expiration
    r = 0.05     # 5% risk-free rate
    sigma = 0.2  # 20% volatility
    
    calc = OptionsCalculator()
    
    # Black-Scholes pricing
    call_price = calc.black_scholes(S, K, T, r, sigma, 'call')
    put_price = calc.black_scholes(S, K, T, r, sigma, 'put')
    
    print(f"Stock Price: ${S}")
    print(f"Strike Price: ${K}")
    print(f"Time to Expiration: {T} years ({T*12:.1f} months)")
    print(f"Risk-free Rate: {r:.1%}")
    print(f"Volatility: {sigma:.1%}")
    print(f"\nBlack-Scholes Prices:")
    print(f"Call Price: ${call_price:.2f}")
    print(f"Put Price: ${put_price:.2f}")
    
    # Put-Call Parity check
    synthetic_call = put_price + S - K * np.exp(-r * T)
    print(f"\nPut-Call Parity Check:")
    print(f"Synthetic Call: ${synthetic_call:.2f}")
    print(f"Difference: ${abs(call_price - synthetic_call):.4f}")
    
    # Binomial tree pricing
    binomial_call = calc.binomial_tree(S, K, T, r, sigma, steps=100, option_type='call')
    binomial_put = calc.binomial_tree(S, K, T, r, sigma, steps=100, option_type='put')
    
    print(f"\nBinomial Tree Prices (100 steps):")
    print(f"Call Price: ${binomial_call:.2f}")
    print(f"Put Price: ${binomial_put:.2f}")
    
    # Monte Carlo pricing
    mc_call, mc_se_call = calc.monte_carlo(S, K, T, r, sigma, simulations=100000, option_type='call')
    mc_put, mc_se_put = calc.monte_carlo(S, K, T, r, sigma, simulations=100000, option_type='put')
    
    print(f"\nMonte Carlo Prices (100,000 simulations):")
    print(f"Call Price: ${mc_call:.2f} ± ${mc_se_call:.2f}")
    print(f"Put Price: ${mc_put:.2f} ± ${mc_se_put:.2f}")
    
    return call_price, put_price


def greeks_analysis():
    """Demonstrate Greeks calculations."""
    print("\n=== Greeks Analysis ===")
    
    S = 100      # Current stock price
    K = 100      # At-the-money
    T = 0.25     # 3 months to expiration
    r = 0.05     # 5% risk-free rate
    sigma = 0.2  # 20% volatility
    
    calc = OptionsCalculator()
    
    # Calculate Greeks for ATM options
    call_greeks = calc.calculate_greeks(S, K, T, r, sigma, 'call')
    put_greeks = calc.calculate_greeks(S, K, T, r, sigma, 'put')
    
    print(f"At-The-Money Options (S=${S}, K=${K}):")
    print(f"{'Greek':<10} {'Call':<10} {'Put':<10}")
    print("-" * 30)
    
    for greek in call_greeks.keys():
        print(f"{greek.capitalize():<10} {call_greeks[greek]:<10.4f} {put_greeks[greek]:<10.4f}")
    
    # Greeks across different strikes
    strikes = range(90, 111, 2)
    print(f"\n{'Strike':<8} {'Call Delta':<12} {'Call Gamma':<12} {'Call Theta':<12} {'Call Vega':<12}")
    print("-" * 56)
    
    for strike in strikes:
        greeks = calc.calculate_greeks(S, strike, T, r, sigma, 'call')
        print(f"{strike:<8} {greeks['delta']:<12.4f} {greeks['gamma']:<12.4f} "
              f"{greeks['theta']:<12.4f} {greeks['vega']:<12.4f}")


def implied_volatility_analysis():
    """Demonstrate implied volatility calculations."""
    print("\n=== Implied Volatility Analysis ===")
    
    S = 100      # Current stock price
    K = 100      # Strike price
    T = 0.25     # 3 months to expiration
    r = 0.05     # 5% risk-free rate
    
    calc = OptionsCalculator()
    
    # Create theoretical option prices with different volatilities
    volatilities = [0.15, 0.20, 0.25, 0.30, 0.35]
    
    print(f"{'True Vol':<10} {'Option Price':<15} {'Implied Vol':<15} {'Error':<10}")
    print("-" * 50)
    
    for true_vol in volatilities:
        # Calculate theoretical price
        theoretical_price = calc.black_scholes(S, K, T, r, true_vol, 'call')
        
        # Calculate implied volatility from the price
        implied_vol = calc.implied_volatility(theoretical_price, S, K, T, r, 'call')
        
        error = abs(true_vol - implied_vol)
        
        print(f"{true_vol:<10.1%} ${theoretical_price:<14.2f} {implied_vol:<15.1%} {error:<10.6f}")


def portfolio_analysis():
    """Demonstrate option portfolio analysis."""
    print("\n=== Option Portfolio Analysis ===")
    
    # Current market conditions
    S = 100      # Current stock price
    r = 0.05     # Risk-free rate
    sigma = 0.2  # Volatility
    T = 0.25     # 3 months to expiration
    
    # Create a portfolio (Iron Condor strategy)
    portfolio = OptionPortfolio()
    
    # Iron Condor: Sell ATM call and put, buy OTM call and put
    portfolio.add_position(S, 95, T, r, sigma, 'put', -10)    # Short 10 puts at 95
    portfolio.add_position(S, 100, T, r, sigma, 'call', -10)  # Short 10 calls at 100
    portfolio.add_position(S, 90, T, r, sigma, 'put', 10)     # Long 10 puts at 90
    portfolio.add_position(S, 105, T, r, sigma, 'call', 10)   # Long 10 calls at 105
    
    print("Iron Condor Portfolio:")
    print("- Short 10 x 95 Puts")
    print("- Short 10 x 100 Calls")
    print("- Long 10 x 90 Puts")
    print("- Long 10 x 105 Calls")
    
    # Calculate portfolio Greeks
    portfolio_greeks = portfolio.calculate_portfolio_greeks()
    
    print(f"\nPortfolio Greeks:")
    for greek, value in portfolio_greeks.items():
        print(f"{greek.capitalize()}: {value:.2f}")
    
    # P&L analysis across different stock prices
    stock_prices = range(85, 116, 2)
    
    print(f"\n{'Stock Price':<12} {'P&L':<12} {'Delta':<12}")
    print("-" * 36)
    
    for price in stock_prices:
        pnl = portfolio.portfolio_pnl(price)
        
        # Calculate portfolio delta at this price
        temp_portfolio = OptionPortfolio()
        temp_portfolio.add_position(price, 95, T, r, sigma, 'put', -10)
        temp_portfolio.add_position(price, 100, T, r, sigma, 'call', -10)
        temp_portfolio.add_position(price, 90, T, r, sigma, 'put', 10)
        temp_portfolio.add_position(price, 105, T, r, sigma, 'call', 10)
        temp_greeks = temp_portfolio.calculate_portfolio_greeks()
        
        print(f"${price:<11} ${pnl:<11.2f} {temp_greeks['delta']:<11.2f}")


def real_options_data_example():
    """Example using real market data for options analysis."""
    print("\n=== Real Options Data Example ===")
    
    try:
        scraper = DataScraper()
        
        # Get options data for a stock (e.g., AAPL)
        symbol = "AAPL"
        print(f"Fetching options data for {symbol}...")
        
        options_data = scraper.get_options_data(symbol)
        
        if not options_data:
            print("No options data available")
            return
        
        calls = options_data['calls']
        puts = options_data['puts']
        exp_date = options_data['expiration_date']
        
        print(f"Expiration Date: {exp_date}")
        print(f"Available Calls: {len(calls)}")
        print(f"Available Puts: {len(puts)}")
        
        # Show some call options
        if not calls.empty:
            print(f"\nSample Call Options:")
            print(calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].head())
        
        # Show some put options
        if not puts.empty:
            print(f"\nSample Put Options:")
            print(puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].head())
        
        # Get stock info for current price
        stock_info = scraper.get_stock_info(symbol)
        if stock_info and 'currentPrice' in stock_info:
            current_price = stock_info['currentPrice']
            print(f"\nCurrent Stock Price: ${current_price:.2f}")
            
            # Find ATM options for analysis
            if not calls.empty:
                atm_calls = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:3]]
                print(f"\nNear-ATM Calls:")
                for _, row in atm_calls.iterrows():
                    strike = row['strike']
                    market_price = row['lastPrice']
                    iv = row['impliedVolatility']
                    
                    print(f"Strike ${strike}: Market Price ${market_price:.2f}, IV {iv:.1%}")
    
    except Exception as e:
        print(f"Error fetching real options data: {e}")
        print("This might be due to network issues or API limitations")


if __name__ == "__main__":
    # Run all examples
    try:
        basic_options_pricing()
        print("\n" + "="*60)
        
        greeks_analysis()
        print("\n" + "="*60)
        
        implied_volatility_analysis()
        print("\n" + "="*60)
        
        portfolio_analysis()
        print("\n" + "="*60)
        
        real_options_data_example()
        
    except Exception as e:
        print(f"Error running examples: {e}")
    
    print("\n=== Options Analysis Complete ===")