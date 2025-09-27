"""
Options pricing models including Black-Scholes, Binomial, and Monte Carlo methods.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Union, Tuple, Dict
import math


class OptionsCalculator:
    """Class containing various options pricing models and Greeks calculations."""
    
    @staticmethod
    def black_scholes(S: float, K: float, T: float, r: float, sigma: float, 
                     option_type: str = 'call') -> float:
        """
        Calculate option price using Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            Option price
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return price
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary containing all Greeks
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type.lower() == 'call':
            theta = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta /= 365  # Convert to daily theta
        
        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% change
        
        # Rho
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str = 'call') -> float:
        """
        Calculate implied volatility using Brent's method.
        
        Args:
            market_price: Observed market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            option_type: 'call' or 'put'
        
        Returns:
            Implied volatility
        """
        def objective_function(sigma):
            return OptionsCalculator.black_scholes(S, K, T, r, sigma, option_type) - market_price
        
        try:
            iv = brentq(objective_function, 0.001, 5.0)
            return iv
        except ValueError:
            return np.nan
    
    @staticmethod
    def binomial_tree(S: float, K: float, T: float, r: float, sigma: float,
                     steps: int = 100, option_type: str = 'call', 
                     american: bool = False) -> float:
        """
        Price options using binomial tree model.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            steps: Number of time steps in the tree
            option_type: 'call' or 'put'
            american: True for American options, False for European
        
        Returns:
            Option price
        """
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
        
        # Initialize asset prices at maturity
        S_T = np.array([S * (u ** (steps - i)) * (d ** i) for i in range(steps + 1)])
        
        # Initialize option values at maturity
        if option_type.lower() == 'call':
            option_values = np.maximum(S_T - K, 0)
        else:
            option_values = np.maximum(K - S_T, 0)
        
        # Step backwards through the tree
        for i in range(steps - 1, -1, -1):
            # Calculate option values at each node
            option_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])
            
            # For American options, check early exercise
            if american:
                S_current = np.array([S * (u ** (i - j)) * (d ** j) for j in range(i + 1)])
                if option_type.lower() == 'call':
                    exercise_values = np.maximum(S_current - K, 0)
                else:
                    exercise_values = np.maximum(K - S_current, 0)
                option_values = np.maximum(option_values, exercise_values)
        
        return option_values[0]
    
    @staticmethod
    def monte_carlo(S: float, K: float, T: float, r: float, sigma: float,
                   simulations: int = 10000, option_type: str = 'call') -> Tuple[float, float]:
        """
        Price options using Monte Carlo simulation.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            simulations: Number of Monte Carlo simulations
            option_type: 'call' or 'put'
        
        Returns:
            Tuple of (option_price, standard_error)
        """
        np.random.seed(42)  # For reproducibility
        
        # Generate random price paths
        Z = np.random.standard_normal(simulations)
        S_T = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(simulations)
        
        return option_price, standard_error


class OptionPortfolio:
    """Class for managing and analyzing option portfolios."""
    
    def __init__(self):
        """Initialize empty option portfolio."""
        self.positions = []
    
    def add_position(self, S: float, K: float, T: float, r: float, sigma: float,
                    option_type: str, quantity: int, premium_paid: float = None):
        """
        Add an option position to the portfolio.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            quantity: Number of contracts (positive for long, negative for short)
            premium_paid: Premium paid for the option (if None, uses Black-Scholes)
        """
        if premium_paid is None:
            premium_paid = OptionsCalculator.black_scholes(S, K, T, r, sigma, option_type)
        
        position = {
            'S': S,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'option_type': option_type,
            'quantity': quantity,
            'premium_paid': premium_paid
        }
        self.positions.append(position)
    
    def calculate_portfolio_greeks(self) -> Dict[str, float]:
        """
        Calculate total portfolio Greeks.
        
        Returns:
            Dictionary with portfolio Greeks
        """
        total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        for position in self.positions:
            greeks = OptionsCalculator.calculate_greeks(
                position['S'], position['K'], position['T'],
                position['r'], position['sigma'], position['option_type']
            )
            
            for greek, value in greeks.items():
                total_greeks[greek] += value * position['quantity']
        
        return total_greeks
    
    def portfolio_pnl(self, new_S: float) -> float:
        """
        Calculate portfolio P&L for a new stock price.
        
        Args:
            new_S: New stock price
        
        Returns:
            Portfolio P&L
        """
        total_pnl = 0
        
        for position in self.positions:
            # Current option value
            current_value = OptionsCalculator.black_scholes(
                new_S, position['K'], position['T'],
                position['r'], position['sigma'], position['option_type']
            )
            
            # P&L = (Current Value - Premium Paid) * Quantity
            pnl = (current_value - position['premium_paid']) * position['quantity']
            total_pnl += pnl
        
        return total_pnl


if __name__ == "__main__":
    # Example usage
    calc = OptionsCalculator()
    
    # Example parameters
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 0.25  # 3 months to expiration
    r = 0.05  # 5% risk-free rate
    sigma = 0.2  # 20% volatility
    
    # Calculate option price
    call_price = calc.black_scholes(S, K, T, r, sigma, 'call')
    put_price = calc.black_scholes(S, K, T, r, sigma, 'put')
    
    print(f"Black-Scholes Call Price: ${call_price:.2f}")
    print(f"Black-Scholes Put Price: ${put_price:.2f}")
    
    # Calculate Greeks
    greeks = calc.calculate_greeks(S, K, T, r, sigma, 'call')
    print(f"\nCall Option Greeks:")
    for greek, value in greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")
    
    # Binomial tree pricing
    binomial_price = calc.binomial_tree(S, K, T, r, sigma, steps=100, option_type='call')
    print(f"\nBinomial Tree Call Price: ${binomial_price:.2f}")
    
    # Monte Carlo pricing
    mc_price, mc_se = calc.monte_carlo(S, K, T, r, sigma, simulations=10000, option_type='call')
    print(f"Monte Carlo Call Price: ${mc_price:.2f} ± ${mc_se:.2f}")
    
    # Portfolio example
    portfolio = OptionPortfolio()
    portfolio.add_position(S, K, T, r, sigma, 'call', 10)  # Long 10 calls
    portfolio.add_position(S, K-10, T, r, sigma, 'put', -5)  # Short 5 puts
    
    portfolio_greeks = portfolio.calculate_portfolio_greeks()
    print(f"\nPortfolio Greeks:")
    for greek, value in portfolio_greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")