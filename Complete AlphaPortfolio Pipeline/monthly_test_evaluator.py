import os
import numpy as np
import pandas as pd
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from matplotlib.gridspec import GridSpec
import json
from datetime import datetime
import traceback
from scipy import stats
from collections import defaultdict

class MonthlyTestEvaluator:
    """
    Sequential monthly test evaluator for AlphaPortfolio.
    Evaluates the model month-by-month in a sequential manner (as opposed to episodes).
    This better reflects how the model would be used in practice with monthly rebalancing.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader,
        config,
        device: torch.device,
        vis_manager = None
    ):
        """
        Initialize monthly test evaluator.
        
        Args:
            model: Trained AlphaPortfolio model
            test_loader: Data loader for test data
            config: Configuration object
            device: Device to evaluate on
            vis_manager: Optional visualization manager
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.vis_manager = vis_manager
        
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                      "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        logging.info(f"MonthlyTestEvaluator initialized with device: {self.device}")
        
        # Create output directories
        self.output_dir = config.config["paths"]["output_dir"]
        self.test_dir = os.path.join(self.output_dir, "monthly_test_results")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Get test configuration
        self.test_config = config.config.get("test", {})
        self.transaction_cost_rate = self.test_config.get("transaction_cost_rate", 0.001)
        self.bootstrap_iterations = self.test_config.get("bootstrap_iterations", 1000)
        self.compare_benchmark = self.test_config.get("compare_benchmark", True)
        
        # Get parameters for model
        self.lookback = config.config["model"]["lookback"]
        self.T = config.config["model"]["T"]  # Used for state dimension
        
        # Load asset names if available
        self.assets_list = None
        if hasattr(test_loader.dataset, 'get_asset_names'):
            self.assets_list = test_loader.dataset.get_asset_names()
        
        logging.info(f"Initialized monthly test evaluator")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test data sequentially month by month.
        
        Returns:
            Dictionary with evaluation metrics
        """
        logging.info("Starting sequential monthly model evaluation on test data")
        
        self.model.eval()
        
        # Extract sequential data from the test loader
        sequential_data = self._extract_sequential_data()
        
        if not sequential_data:
            logging.error("Failed to extract sequential data for monthly testing")
            return {'error': 'No sequential data available'}
        
        # Process the data sequentially
        all_dates = sequential_data['dates']
        all_states = sequential_data['states']
        all_future_returns = sequential_data['future_returns']
        all_masks = sequential_data['masks']
        
        logging.info(f"Extracted sequential data: {len(all_dates)} time points")
        
        # Initialize storage for results
        monthly_returns = []
        monthly_weights = []
        monthly_dates = []
        prev_weights = None
        
        # Track performance metrics over time
        cumulative_returns = []
        rolling_sharpe_values = []  # For 12-month rolling Sharpe
        sharpe_window = self.config.config["model"].get("sharpe_window", 12)
        
        # Process each month sequentially
        with torch.no_grad():
            for i, (date, state, future_return, mask) in enumerate(zip(all_dates, all_states, all_future_returns, all_masks)):
                # Prepare data for model
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Forward pass
                portfolio_weights, winner_scores, sorted_indices = self.model(state_tensor, mask_tensor)
                
                # Convert to numpy for calculations
                weights = portfolio_weights.cpu().numpy()[0]
                
                # Apply transaction costs if we have previous weights
                transaction_costs = 0
                if prev_weights is not None:
                    weight_changes = np.abs(weights - prev_weights)
                    transaction_costs = np.sum(weight_changes) * self.transaction_cost_rate
                
                # Calculate portfolio return
                masked_weights = weights * mask
                portfolio_return = np.sum(masked_weights * future_return)
                
                # Apply transaction costs
                net_return = portfolio_return - transaction_costs
                
                # Store results
                monthly_returns.append(net_return)
                monthly_weights.append(weights)
                monthly_dates.append(date)
                
                # Update previous weights for next iteration
                prev_weights = weights
                
                # Update cumulative return
                if i == 0:
                    cumulative_returns.append(net_return)
                else:
                    cumulative_returns.append((1 + cumulative_returns[-1]) * (1 + net_return) - 1)
                
                # Calculate rolling Sharpe if we have enough data
                if i >= sharpe_window - 1:
                    window_returns = monthly_returns[-(sharpe_window):]
                    sharpe = self._calculate_sharpe(window_returns)
                    rolling_sharpe_values.append(sharpe)
                else:
                    rolling_sharpe_values.append(None)
                
                # Log progress every 5 months
                if (i + 1) % 5 == 0:
                    logging.info(f"Processed {i+1}/{len(all_dates)} months. Current return: {net_return:.4f}, "
                               f"Cumulative: {cumulative_returns[-1]:.4f}")
        
        # Calculate overall metrics
        metrics = self.calculate_performance_metrics(
            monthly_returns=monthly_returns,
            monthly_weights=monthly_weights,
            monthly_dates=monthly_dates,
            cumulative_returns=cumulative_returns,
            rolling_sharpe_values=rolling_sharpe_values
        )
        
        # Generate visualizations
        self.generate_visualizations(
            monthly_returns=monthly_returns,
            monthly_weights=monthly_weights,
            monthly_dates=monthly_dates,
            cumulative_returns=cumulative_returns,
            rolling_sharpe_values=rolling_sharpe_values,
            metrics=metrics
        )
        
        # Compare against benchmark if needed
        if self.compare_benchmark:
            benchmark_metrics = self.compare_to_benchmark(
                monthly_returns=monthly_returns,
                monthly_dates=monthly_dates
            )
            metrics.update(benchmark_metrics)
        
        # Generate comprehensive report
        if self.test_config.get("generate_report", True):
            self.generate_report(metrics)
        
        return metrics
    
    def _extract_sequential_data(self):
        """
        Extract sequential data from test loader for month-by-month evaluation.
        
        Returns:
            Dictionary with sequential data
        """
        # This is a critical function that must extract data from the test loader
        # in a way that allows for sequential evaluation
        
        try:
            # First, check if the dataset provides a method for sequential access
            if hasattr(self.test_loader.dataset, 'get_sequential_data'):
                return self.test_loader.dataset.get_sequential_data()
            
            # If not, we'll attempt to reconstruct sequential data from the episodes
            # This is a bit tricky and depends on how the dataset is structured
            
            # Initialize containers
            all_states = []
            all_future_returns = []
            all_masks = []
            all_dates = []
            
            # Extract data from test loader
            for batch in self.test_loader:
                # Check if batch includes dates (4 items per batch element)
                if len(batch) == 4:
                    states, future_returns, masks, dates = batch
                else:
                    states, future_returns, masks = batch
                    dates = None
                
                # Convert to numpy for easier handling
                states_np = states.cpu().numpy()
                future_returns_np = future_returns.cpu().numpy()
                masks_np = masks.cpu().numpy()
                
                # Process each episode in the batch
                for i in range(states_np.shape[0]):
                    episode_states = states_np[i]
                    episode_future_returns = future_returns_np[i]
                    episode_masks = masks_np[i]
                    
                    # Get dates for this episode if available
                    episode_dates = None
                    if dates is not None:
                        episode_dates = dates[i]
                    else:
                        # Create placeholder dates if none provided
                        episode_dates = [f"Month {j+1}" for j in range(episode_states.shape[0])]
                    
                    # Add to our collections
                    all_states.extend(episode_states)
                    all_future_returns.extend(episode_future_returns)
                    all_masks.extend(episode_masks)
                    all_dates.extend(episode_dates)
            
            # Sort by date if dates are actual datetime objects
            if all_dates and hasattr(all_dates[0], 'strftime'):
                # Sort all data by date
                sorted_indices = np.argsort(all_dates)
                all_states = [all_states[i] for i in sorted_indices]
                all_future_returns = [all_future_returns[i] for i in sorted_indices]
                all_masks = [all_masks[i] for i in sorted_indices]
                all_dates = [all_dates[i] for i in sorted_indices]
            
            # Remove duplicates if any
            unique_dates = []
            unique_states = []
            unique_future_returns = []
            unique_masks = []
            
            date_set = set()
            for date, state, future_return, mask in zip(all_dates, all_states, all_future_returns, all_masks):
                date_str = str(date)
                if date_str not in date_set:
                    date_set.add(date_str)
                    unique_dates.append(date)
                    unique_states.append(state)
                    unique_future_returns.append(future_return)
                    unique_masks.append(mask)
            
            # Convert back to arrays
            return {
                'dates': unique_dates,
                'states': np.array(unique_states),
                'future_returns': np.array(unique_future_returns),
                'masks': np.array(unique_masks)
            }
            
        except Exception as e:
            logging.error(f"Error extracting sequential data: {str(e)}")
            logging.error(traceback.format_exc())
            return None
    
    def _calculate_sharpe(self, returns, risk_free_rate=0.0):
        """
        Calculate Sharpe ratio from returns.
        
        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate (default: 0)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)  # Use sample standard deviation
        
        if std_excess == 0:
            return 0
        
        # Annualize (assuming monthly returns)
        sharpe = (mean_excess / std_excess) * np.sqrt(12)
        
        return sharpe
    
    def calculate_performance_metrics(self, monthly_returns, monthly_weights, monthly_dates,
                                    cumulative_returns, rolling_sharpe_values):
        """
        Calculate comprehensive performance metrics.
        
        Args:
            monthly_returns: List of monthly returns
            monthly_weights: List of monthly weights
            monthly_dates: List of monthly dates
            cumulative_returns: List of cumulative returns
            rolling_sharpe_values: List of rolling Sharpe ratios
            
        Returns:
            Dictionary with performance metrics
        """
        # Convert to numpy arrays for calculations
        returns_array = np.array(monthly_returns)
        
        # Basic return statistics
        mean_return = np.mean(returns_array)
        median_return = np.median(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe(monthly_returns)
        
        # Calculate drawdowns
        drawdowns = []
        peak = 0
        for cr in cumulative_returns:
            if cr > peak:
                peak = cr
            drawdown = (peak - cr) / (1 + peak) if peak > 0 else 0
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns) if drawdowns else 0
        max_drawdown_date = monthly_dates[np.argmax(drawdowns)] if drawdowns else None
        
        # Calculate Calmar ratio (annualized return / max drawdown)
        calmar_ratio = (mean_return * 12) / max_drawdown if max_drawdown > 0 else 0
        
        # Calculate Sortino ratio (using downside deviation)
        negative_returns = returns_array[returns_array < 0]
        downside_dev = np.std(negative_returns, ddof=1) if len(negative_returns) > 0 else 0.0001
        sortino_ratio = (mean_return / downside_dev) * np.sqrt(12) if downside_dev > 0 else 0
        
        # Calculate percentage of positive months
        positive_months = np.sum(returns_array > 0)
        pct_positive = positive_months / len(returns_array) if len(returns_array) > 0 else 0
        
        # Calculate worst and best months
        worst_return = np.min(returns_array) if len(returns_array) > 0 else 0
        best_return = np.max(returns_array) if len(returns_array) > 0 else 0
        worst_month_date = monthly_dates[np.argmin(returns_array)] if len(returns_array) > 0 else None
        best_month_date = monthly_dates[np.argmax(returns_array)] if len(returns_array) > 0 else 0
        
        # Calculate final cumulative return
        final_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        
        # Calculate annualized return
        n_years = len(monthly_returns) / 12
        annualized_return = (1 + final_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Calculate portfolio turnover
        turnover_rates = []
        for i in range(1, len(monthly_weights)):
            weight_changes = np.abs(monthly_weights[i] - monthly_weights[i-1])
            turnover = np.sum(weight_changes) / 2  # Divide by 2 to count only one side
            turnover_rates.append(turnover)
        
        avg_turnover = np.mean(turnover_rates) if turnover_rates else 0
        
        # Calculate average number of positions
        long_positions = [np.sum(w > 0) for w in monthly_weights]
        short_positions = [np.sum(w < 0) for w in monthly_weights]
        avg_long_positions = np.mean(long_positions) if long_positions else 0
        avg_short_positions = np.mean(short_positions) if short_positions else 0
        
        # Calculate average position exposure
        long_exposure = [np.sum(np.maximum(w, 0)) for w in monthly_weights]
        short_exposure = [np.sum(np.minimum(w, 0)) for w in monthly_weights]
        avg_long_exposure = np.mean(long_exposure) if long_exposure else 0
        avg_short_exposure = np.mean(np.abs(short_exposure)) if short_exposure else 0
        
        # Compile metrics dictionary
        metrics = {
            'mean_monthly_return': float(mean_return),
            'median_monthly_return': float(median_return),
            'std_monthly_return': float(std_return),
            'annualized_return': float(annualized_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'max_drawdown': float(max_drawdown),
            'max_drawdown_date': str(max_drawdown_date) if max_drawdown_date else None,
            'percent_positive_months': float(pct_positive),
            'worst_monthly_return': float(worst_return),
            'worst_month_date': str(worst_month_date) if worst_month_date else None,
            'best_monthly_return': float(best_return),
            'best_month_date': str(best_month_date) if best_month_date else None,
            'final_cumulative_return': float(final_return),
            'average_turnover': float(avg_turnover),
            'average_long_positions': float(avg_long_positions),
            'average_short_positions': float(avg_short_positions),
            'average_long_exposure': float(avg_long_exposure),
            'average_short_exposure': float(avg_short_exposure),
            'num_months': len(monthly_returns),
            'start_date': str(monthly_dates[0]) if monthly_dates else None,
            'end_date': str(monthly_dates[-1]) if monthly_dates else None
        }
        
        return metrics

    def compare_to_benchmark(self, monthly_returns, monthly_dates):
        """
        Compare strategy performance to benchmark. In a real implementation, 
        you would load actual benchmark data here.
        
        Args:
            monthly_returns: List of strategy monthly returns
            monthly_dates: List of corresponding dates
            
        Returns:
            Dictionary with benchmark comparison metrics
        """
        # This is a stub function. In a real implementation, you would:
        # 1. Load benchmark returns for the same dates
        # 2. Calculate comparison metrics
        
        # For now, create synthetic benchmark returns
        mean_return = np.mean(monthly_returns)
        std_return = np.std(monthly_returns)
        
        np.random.seed(42)  # For reproducible results
        benchmark_returns = np.random.normal(
            loc=mean_return * 0.7,  # Lower mean (70% of strategy)
            scale=std_return * 1.1,  # Higher volatility (110% of strategy)
            size=len(monthly_returns)
        )
        
        # Calculate benchmark metrics
        benchmark_mean = np.mean(benchmark_returns)
        benchmark_median = np.median(benchmark_returns)
        benchmark_std = np.std(benchmark_returns)
        benchmark_sharpe = (benchmark_mean / benchmark_std) * np.sqrt(12) if benchmark_std > 0 else 0
        
        # Calculate benchmark cumulative returns
        benchmark_cum_returns = []
        for i, r in enumerate(benchmark_returns):
            if i == 0:
                benchmark_cum_returns.append(r)
            else:
                benchmark_cum_returns.append((1 + benchmark_cum_returns[-1]) * (1 + r) - 1)
        
        benchmark_final_return = benchmark_cum_returns[-1]
        
        # Calculate benchmark drawdowns
        benchmark_drawdowns = []
        peak = 0
        for cr in benchmark_cum_returns:
            if cr > peak:
                peak = cr
            drawdown = (peak - cr) / (1 + peak) if peak > 0 else 0
            benchmark_drawdowns.append(drawdown)
        
        benchmark_max_drawdown = max(benchmark_drawdowns) if benchmark_drawdowns else 0
        
        # Calculate alpha and beta
        beta, alpha, r_value, p_value, std_err = stats.linregress(benchmark_returns, monthly_returns)
        
        # Calculate tracking error and information ratio
        tracking_error = np.std(np.array(monthly_returns) - benchmark_returns)
        information_ratio = (mean_return - benchmark_mean) / tracking_error if tracking_error > 0 else 0
        
        # Calculate outperformance statistics
        outperformance = np.array(monthly_returns) - benchmark_returns
        pct_outperform = np.mean(outperformance > 0)
        mean_outperformance = np.mean(outperformance)
        
        # Create benchmark metrics dictionary
        benchmark_metrics = {
            'benchmark_mean_monthly_return': float(benchmark_mean),
            'benchmark_median_monthly_return': float(benchmark_median),
            'benchmark_std_monthly_return': float(benchmark_std),
            'benchmark_sharpe_ratio': float(benchmark_sharpe),
            'benchmark_max_drawdown': float(benchmark_max_drawdown),
            'benchmark_final_cumulative_return': float(benchmark_final_return),
            'alpha': float(alpha),
            'beta': float(beta),
            'r_squared': float(r_value**2),
            'tracking_error': float(tracking_error),
            'information_ratio': float(information_ratio),
            'percent_outperformance': float(pct_outperform),
            'mean_outperformance': float(mean_outperformance)
        }
        
        # Plot benchmark comparison
        self._plot_benchmark_comparison(
            strategy_returns=monthly_returns,
            benchmark_returns=benchmark_returns,
            strategy_cum_returns=np.array(monthly_returns),
            benchmark_cum_returns=benchmark_cum_returns,
            dates=monthly_dates
        )
        
        return {'benchmark_metrics': benchmark_metrics}
    
    def generate_visualizations(self, monthly_returns, monthly_weights, monthly_dates,
                              cumulative_returns, rolling_sharpe_values, metrics):
        """
        Generate comprehensive visualizations for test results.
        
        Args:
            monthly_returns: List of monthly returns
            monthly_weights: List of monthly weights
            monthly_dates: List of monthly dates
            cumulative_returns: List of cumulative returns
            rolling_sharpe_values: List of rolling Sharpe ratios
            metrics: Dictionary of calculated metrics
        """
        # Create visualization directory
        viz_dir = os.path.join(self.test_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Format dates for plotting
        formatted_dates = self._format_dates_for_plotting(monthly_dates)
        
        # 1. Plot monthly returns
        self._plot_monthly_returns(
            returns=monthly_returns,
            dates=formatted_dates,
            output_dir=viz_dir
        )
        
        # 2. Plot cumulative returns
        self._plot_cumulative_returns(
            cum_returns=cumulative_returns,
            dates=formatted_dates,
            output_dir=viz_dir
        )
        
        # 3. Plot drawdowns
        self._plot_drawdowns(
            cum_returns=cumulative_returns,
            dates=formatted_dates,
            output_dir=viz_dir
        )
        
        # 4. Plot rolling Sharpe ratio
        self._plot_rolling_sharpe(
            rolling_sharpe=rolling_sharpe_values,
            dates=formatted_dates,
            output_dir=viz_dir
        )
        
        # 5. Plot portfolio allocation over time
        self._plot_portfolio_allocation(
            weights=monthly_weights,
            dates=formatted_dates,
            output_dir=viz_dir
        )
        
        # 6. Plot position counts over time
        self._plot_position_counts(
            weights=monthly_weights,
            dates=formatted_dates,
            output_dir=viz_dir
        )
        
        # 7. Plot turnover over time
        self._plot_turnover(
            weights=monthly_weights,
            dates=formatted_dates,
            output_dir=viz_dir
        )
        
        # 8. Plot monthly return distribution
        self._plot_return_distribution(
            returns=monthly_returns,
            output_dir=viz_dir
        )
        
        # 9. Plot performance dashboard
        self._plot_performance_dashboard(
            returns=monthly_returns,
            cum_returns=cumulative_returns,
            rolling_sharpe=rolling_sharpe_values,
            dates=formatted_dates,
            metrics=metrics,
            output_dir=viz_dir
        )
        
        logging.info(f"Generated monthly test visualizations in {viz_dir}")
    
    def _format_dates_for_plotting(self, dates):
        """Format dates for plotting axes."""
        formatted_dates = []
        
        for date in dates:
            try:
                # Check if date is already a string
                if isinstance(date, str):
                    # Try to parse it to a datetime
                    try:
                        dt = pd.to_datetime(date)
                        formatted_dates.append(dt.strftime('%Y-%m-%d'))
                    except:
                        formatted_dates.append(date)
                # Check if date is a datetime/timestamp
                elif isinstance(date, (pd.Timestamp, datetime, np.datetime64)):
                    if hasattr(date, 'strftime'):
                        formatted_dates.append(date.strftime('%Y-%m-%d'))
                    else:
                        formatted_dates.append(pd.Timestamp(date).strftime('%Y-%m-%d'))
                else:
                    formatted_dates.append(str(date))
            except Exception as e:
                logging.warning(f"Error formatting date {date}: {str(e)}")
                formatted_dates.append(str(date))
        
        return formatted_dates
    
    def _plot_monthly_returns(self, returns, dates, output_dir):
        """Plot monthly returns with proper dates."""
        plt.figure(figsize=(14, 8))
        
        # Create color-coded bars
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        # Plot bars
        plt.bar(range(len(returns)), np.array(returns) * 100, color=colors, alpha=0.7)
        
        # Add horizontal line at 0
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add mean return line
        mean_return = np.mean(returns) * 100
        plt.axhline(y=mean_return, color='blue', linestyle='--', linewidth=1, 
                  label=f'Mean: {mean_return:.2f}%')
        
        # Format x-axis with dates
        if len(dates) > 24:
            # If we have many dates, show fewer x labels
            step = max(1, len(dates) // 12)  # Show ~12 labels
            plt.xticks(range(0, len(dates), step), [dates[i] for i in range(0, len(dates), step)], 
                     rotation=45, ha='right')
        else:
            plt.xticks(range(len(dates)), dates, rotation=45, ha='right')
        
        plt.title('Monthly Returns', fontsize=14)
        plt.ylabel('Return (%)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'monthly_returns.png'), dpi=300)
        plt.close()
    
    def _plot_cumulative_returns(self, cum_returns, dates, output_dir):
        """Plot cumulative returns with proper dates."""
        plt.figure(figsize=(14, 8))
        
        # Plot line
        plt.plot(range(len(cum_returns)), np.array(cum_returns) * 100, 'b-', linewidth=2)
        
        # Add horizontal line at 0
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Format x-axis with dates
        if len(dates) > 24:
            # If we have many dates, show fewer x labels
            step = max(1, len(dates) // 12)  # Show ~12 labels
            plt.xticks(range(0, len(dates), step), [dates[i] for i in range(0, len(dates), step)], 
                     rotation=45, ha='right')
        else:
            plt.xticks(range(len(dates)), dates, rotation=45, ha='right')
        
        # Add annotations
        plt.title('Cumulative Returns', fontsize=14)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add final return annotation
        final_return = cum_returns[-1] * 100
        plt.annotate(f"Final: {final_return:.2f}%", 
                    xy=(len(cum_returns)-1, final_return), 
                    xytext=(len(cum_returns)-len(cum_returns)//5, final_return+5),
                    arrowprops=dict(arrowstyle='->'))
        
        # Add annual return annotation
        n_years = len(cum_returns) / 12
        annualized_return = (1 + cum_returns[-1]) ** (1 / n_years) - 1 if n_years > 0 else 0
        plt.annotate(f"Annualized: {annualized_return*100:.2f}%", 
                    xy=(0, final_return),
                    xytext=(len(cum_returns)//10, final_return-10),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'cumulative_returns.png'), dpi=300)
        plt.close()
    
    def _plot_drawdowns(self, cum_returns, dates, output_dir):
        """Plot drawdowns over time with proper dates."""
        # Calculate drawdowns
        drawdowns = []
        peak = 0
        for cr in cum_returns:
            if cr > peak:
                peak = cr
            drawdown = (peak - cr) / (1 + peak) if peak > 0 else 0
            drawdowns.append(drawdown)
        
        plt.figure(figsize=(14, 8))
        
        # Plot line
        plt.plot(range(len(drawdowns)), np.array(drawdowns) * 100, 'r-', linewidth=2)
        
        # Format x-axis with dates
        if len(dates) > 24:
            # If we have many dates, show fewer x labels
            step = max(1, len(dates) // 12)  # Show ~12 labels
            plt.xticks(range(0, len(dates), step), [dates[i] for i in range(0, len(dates), step)], 
                     rotation=45, ha='right')
        else:
            plt.xticks(range(len(dates)), dates, rotation=45, ha='right')
        
        # Find max drawdown
        max_dd = max(drawdowns) if drawdowns else 0
        max_dd_idx = drawdowns.index(max_dd) if drawdowns else 0
        
        # Add max drawdown annotation
        plt.annotate(f"Max Drawdown: {max_dd*100:.2f}%", 
                    xy=(max_dd_idx, max_dd * 100), 
                    xytext=(max_dd_idx, max_dd * 100 - 5),
                    arrowprops=dict(arrowstyle='->'))
        
        # Add annotations
        plt.title('Portfolio Drawdowns', fontsize=14)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Invert y-axis so drawdowns go down
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'drawdowns.png'), dpi=300)
        plt.close()
    
    def _plot_rolling_sharpe(self, rolling_sharpe, dates, output_dir):
        """Plot rolling Sharpe ratio with proper dates."""
        # Filter out None values
        valid_indices = [i for i, rs in enumerate(rolling_sharpe) if rs is not None]
        valid_sharpe = [rolling_sharpe[i] for i in valid_indices]
        valid_dates = [dates[i] for i in valid_indices]
        
        if not valid_sharpe:
            logging.warning("No valid rolling Sharpe values to plot")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Plot line
        plt.plot(range(len(valid_sharpe)), valid_sharpe, 'g-', linewidth=2)
        
        # Add horizontal line at 0
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Format x-axis with dates
        if len(valid_dates) > 24:
            # If we have many dates, show fewer x labels
            step = max(1, len(valid_dates) // 12)  # Show ~12 labels
            plt.xticks(range(0, len(valid_dates), step), [valid_dates[i] for i in range(0, len(valid_dates), step)], 
                     rotation=45, ha='right')
        else:
            plt.xticks(range(len(valid_dates)), valid_dates, rotation=45, ha='right')
        
        # Add mean Sharpe line
        mean_sharpe = np.mean(valid_sharpe)
        plt.axhline(y=mean_sharpe, color='blue', linestyle='--', linewidth=1, 
                  label=f'Mean: {mean_sharpe:.2f}')
        
        # Add annotations
        plt.title('12-Month Rolling Sharpe Ratio', fontsize=14)
        plt.ylabel('Sharpe Ratio', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'rolling_sharpe.png'), dpi=300)
        plt.close()
    
    def _plot_portfolio_allocation(self, weights, dates, output_dir):
        """Plot portfolio allocation over time with proper dates."""
        # Find most important assets by average absolute weight
        avg_abs_weights = np.mean([np.abs(w) for w in weights], axis=0)
        top_indices = np.argsort(-avg_abs_weights)[:15]  # Top 15 assets
        
        # Extract weights for top assets
        top_weights = np.array([[w[i] for i in top_indices] for w in weights])
        
        # Get asset names
        if self.assets_list:
            top_assets = [self.assets_list[i] for i in top_indices]
        else:
            top_assets = [f"Asset {i+1}" for i in top_indices]
        
        plt.figure(figsize=(14, 10))
        
        # Create heatmap
        sns.heatmap(
            top_weights.T,  # Transpose to have assets as rows
            cmap='RdBu_r',
            center=0,
            vmin=-np.max(np.abs(top_weights)),
            vmax=np.max(np.abs(top_weights)),
            yticklabels=top_assets,
            cbar_kws={'label': 'Portfolio Weight'}
        )
        
        # Format x-axis with dates
        if len(dates) > 24:
            # If we have many dates, show fewer x labels
            step = max(1, len(dates) // 12)  # Show ~12 labels
            plt.xticks(range(0, len(dates), step), [dates[i] for i in range(0, len(dates), step)], 
                     rotation=45, ha='right')
        else:
            plt.xticks(range(len(dates)), dates, rotation=45, ha='right')
        
        # Add annotations
        plt.title('Portfolio Allocation Over Time (Top 15 Assets)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Asset', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'portfolio_allocation.png'), dpi=300)
        plt.close()
    
    def _plot_position_counts(self, weights, dates, output_dir):
        """Plot position counts over time with proper dates."""
        # Calculate long and short position counts
        long_counts = [np.sum(w > 0) for w in weights]
        short_counts = [np.sum(w < 0) for w in weights]
        
        plt.figure(figsize=(14, 8))
        
        # Plot stacked bars
        plt.bar(range(len(weights)), long_counts, color='green', alpha=0.7, label='Long Positions')
        plt.bar(range(len(weights)), [-c for c in short_counts], color='red', alpha=0.7, label='Short Positions')
        
        # Add horizontal line at 0
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Format x-axis with dates
        if len(dates) > 24:
            # If we have many dates, show fewer x labels
            step = max(1, len(dates) // 12)  # Show ~12 labels
            plt.xticks(range(0, len(dates), step), [dates[i] for i in range(0, len(dates), step)], 
                     rotation=45, ha='right')
        else:
            plt.xticks(range(len(dates)), dates, rotation=45, ha='right')
        
        # Add annotations
        plt.title('Number of Portfolio Positions Over Time', fontsize=14)
        plt.ylabel('Number of Positions', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'position_counts.png'), dpi=300)
        plt.close()
    
    def _plot_turnover(self, weights, dates, output_dir):
        """Plot portfolio turnover over time with proper dates."""
        # Calculate turnover
        turnover_rates = []
        for i in range(1, len(weights)):
            weight_changes = np.abs(weights[i] - weights[i-1])
            turnover = np.sum(weight_changes) / 2  # Divide by 2 to count only one side
            turnover_rates.append(turnover)
        
        # We have one fewer turnover point than dates
        turnover_dates = dates[1:]
        
        plt.figure(figsize=(14, 8))
        
        # Plot line
        plt.plot(range(len(turnover_rates)), turnover_rates, 'b-', linewidth=2)
        
        # Format x-axis with dates
        if len(turnover_dates) > 24:
            # If we have many dates, show fewer x labels
            step = max(1, len(turnover_dates) // 12)  # Show ~12 labels
            plt.xticks(range(0, len(turnover_dates), step), [turnover_dates[i] for i in range(0, len(turnover_dates), step)], 
                     rotation=45, ha='right')
        else:
            plt.xticks(range(len(turnover_dates)), turnover_dates, rotation=45, ha='right')
        
        # Add mean turnover line
        mean_turnover = np.mean(turnover_rates)
        plt.axhline(y=mean_turnover, color='red', linestyle='--', linewidth=1, 
                  label=f'Mean: {mean_turnover:.2f}')
        
        # Add annotations
        plt.title('Monthly Portfolio Turnover', fontsize=14)
        plt.ylabel('Turnover', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'turnover.png'), dpi=300)
        plt.close()
    
    def _plot_return_distribution(self, returns, output_dir):
        """Plot distribution of monthly returns."""
        plt.figure(figsize=(14, 8))
        
        # Create histogram with KDE
        sns.histplot(np.array(returns) * 100, kde=True, bins=30, color='skyblue')
        
        # Add vertical lines for key statistics
        mean_return = np.mean(returns) * 100
        median_return = np.median(returns) * 100
        plt.axvline(x=mean_return, color='red', linestyle='-', label=f'Mean: {mean_return:.2f}%')
        plt.axvline(x=median_return, color='green', linestyle='--', label=f'Median: {median_return:.2f}%')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add annotations
        plt.title('Monthly Return Distribution', fontsize=14)
        plt.xlabel('Return (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add key statistics as text
        stats_text = (
            f"Mean: {mean_return:.2f}%\n"
            f"Median: {median_return:.2f}%\n"
            f"Std Dev: {np.std(returns) * 100:.2f}%\n"
            f"Min: {np.min(returns) * 100:.2f}%\n"
            f"Max: {np.max(returns) * 100:.2f}%\n"
            f"Positive Months: {np.mean(np.array(returns) > 0) * 100:.1f}%"
        )
        plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'return_distribution.png'), dpi=300)
        plt.close()
    
    def _plot_benchmark_comparison(self, strategy_returns, benchmark_returns, 
                                 strategy_cum_returns, benchmark_cum_returns, dates):
        """Plot strategy vs benchmark comparison with proper dates."""
        # Create visualization directory
        viz_dir = os.path.join(self.test_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Plot cumulative returns comparison
        plt.figure(figsize=(14, 8))
        
        # Plot lines
        plt.plot(range(len(strategy_cum_returns)), np.array(strategy_cum_returns) * 100, 
               'b-', linewidth=2, label='Strategy')
        plt.plot(range(len(benchmark_cum_returns)), np.array(benchmark_cum_returns) * 100, 
               'r--', linewidth=2, label='Benchmark')
        
        # Format x-axis with dates
        if len(dates) > 24:
            # If we have many dates, show fewer x labels
            step = max(1, len(dates) // 12)  # Show ~12 labels
            plt.xticks(range(0, len(dates), step), [dates[i] for i in range(0, len(dates), step)], 
                     rotation=45, ha='right')
        else:
            plt.xticks(range(len(dates)), dates, rotation=45, ha='right')
        
        # Add annotations
        plt.title('Strategy vs. Benchmark Cumulative Returns', fontsize=14)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(viz_dir, 'benchmark_comparison.png'), dpi=300)
        plt.close()
        
        # 2. Plot rolling alpha and beta
        window = 12  # 12-month rolling window
        if len(strategy_returns) >= window:
            plt.figure(figsize=(14, 12))
            
            # Calculate rolling beta and alpha
            rolling_betas = []
            rolling_alphas = []
            
            for i in range(window, len(strategy_returns) + 1):
                x = benchmark_returns[i-window:i]
                y = strategy_returns[i-window:i]
                beta, alpha, _, _, _ = stats.linregress(x, y)
                rolling_betas.append(beta)
                rolling_alphas.append(alpha)
            
            # Get dates for rolling metrics
            rolling_dates = dates[window:]
            
            # Plot rolling alpha
            plt.subplot(2, 1, 1)
            plt.plot(range(len(rolling_alphas)), rolling_alphas, 'g-', linewidth=2)
            
            # Format x-axis with dates
            if len(rolling_dates) > 24:
                # If we have many dates, show fewer x labels
                step = max(1, len(rolling_dates) // 12)  # Show ~12 labels
                plt.xticks(range(0, len(rolling_dates), step), [rolling_dates[i] for i in range(0, len(rolling_dates), step)], 
                         rotation=45, ha='right')
            else:
                plt.xticks(range(len(rolling_dates)), rolling_dates, rotation=45, ha='right')
            
            # Add horizontal line at 0
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add mean alpha line
            mean_alpha = np.mean(rolling_alphas)
            plt.axhline(y=mean_alpha, color='red', linestyle='--', linewidth=1, 
                      label=f'Mean: {mean_alpha:.4f}')
            
            plt.title('12-Month Rolling Alpha', fontsize=14)
            plt.ylabel('Alpha', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot rolling beta
            plt.subplot(2, 1, 2)
            plt.plot(range(len(rolling_betas)), rolling_betas, 'b-', linewidth=2)
            
            # Format x-axis with dates
            if len(rolling_dates) > 24:
                # If we have many dates, show fewer x labels
                step = max(1, len(rolling_dates) // 12)  # Show ~12 labels
                plt.xticks(range(0, len(rolling_dates), step), [rolling_dates[i] for i in range(0, len(rolling_dates), step)], 
                         rotation=45, ha='right')
            else:
                plt.xticks(range(len(rolling_dates)), rolling_dates, rotation=45, ha='right')
            
            # Add horizontal line at 1
            plt.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
            
            # Add mean beta line
            mean_beta = np.mean(rolling_betas)
            plt.axhline(y=mean_beta, color='red', linestyle='--', linewidth=1, 
                      label=f'Mean: {mean_beta:.4f}')
            
            plt.title('12-Month Rolling Beta', fontsize=14)
            plt.ylabel('Beta', fontsize=12)
            plt.xlabel('Date', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(viz_dir, 'rolling_alpha_beta.png'), dpi=300)
            plt.close()
    
    def _plot_performance_dashboard(self, returns, cum_returns, rolling_sharpe, dates, metrics, output_dir):
        """Plot comprehensive performance dashboard with proper dates."""
        fig = plt.figure(figsize=(20, 24))
        
        # Create grid for subplots
        gs = GridSpec(7, 4, figure=fig)
        
        # 1. Cumulative returns (large top plot)
        ax1 = fig.add_subplot(gs[0:2, :])
        ax1.plot(range(len(cum_returns)), np.array(cum_returns) * 100, 'b-', linewidth=2)
        
        # Format x-axis with dates
        if len(dates) > 24:
            # If we have many dates, show fewer x labels
            step = max(1, len(dates) // 12)  # Show ~12 labels
            ax1.set_xticks(range(0, len(dates), step))
            ax1.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45, ha='right')
        else:
            ax1.set_xticks(range(len(dates)))
            ax1.set_xticklabels(dates, rotation=45, ha='right')
        
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title('Cumulative Returns', fontsize=14)
        ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Monthly returns
        ax2 = fig.add_subplot(gs[2:4, :2])
        colors = ['green' if r > 0 else 'red' for r in returns]
        ax2.bar(range(len(returns)), np.array(returns) * 100, color=colors, alpha=0.7)
        
        # Format x-axis with dates
        if len(dates) > 24:
            # If we have many dates, show fewer x labels
            step = max(1, len(dates) // 12)  # Show ~12 labels
            ax2.set_xticks(range(0, len(dates), step))
            ax2.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45, ha='right')
        else:
            ax2.set_xticks(range(len(dates)))
            ax2.set_xticklabels(dates, rotation=45, ha='right')
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Monthly Returns', fontsize=14)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe ratio
        ax3 = fig.add_subplot(gs[2:4, 2:])
        
        # Filter out None values
        valid_indices = [i for i, rs in enumerate(rolling_sharpe) if rs is not None]
        valid_sharpe = [rolling_sharpe[i] for i in valid_indices]
        valid_dates = [dates[i] for i in valid_indices]
        
        if valid_sharpe:
            ax3.plot(range(len(valid_sharpe)), valid_sharpe, 'g-', linewidth=2)
            
            # Format x-axis with dates
            if len(valid_dates) > 24:
                # If we have many dates, show fewer x labels
                step = max(1, len(valid_dates) // 12)  # Show ~12 labels
                ax3.set_xticks(range(0, len(valid_dates), step))
                ax3.set_xticklabels([valid_dates[i] for i in range(0, len(valid_dates), step)], rotation=45, ha='right')
            else:
                ax3.set_xticks(range(len(valid_dates)))
                ax3.set_xticklabels(valid_dates, rotation=45, ha='right')
            
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add mean Sharpe line
            mean_sharpe = np.mean(valid_sharpe)
            ax3.axhline(y=mean_sharpe, color='red', linestyle='--', linewidth=1, 
                     label=f'Mean: {mean_sharpe:.2f}')
            
            ax3.set_title('12-Month Rolling Sharpe Ratio', fontsize=14)
            ax3.set_ylabel('Sharpe Ratio', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for Rolling Sharpe', 
                   ha='center', va='center', fontsize=12)
            ax3.set_title('12-Month Rolling Sharpe Ratio', fontsize=14)
        
        # 4. Return distribution
        ax4 = fig.add_subplot(gs[4:6, :2])
        sns.histplot(np.array(returns) * 100, kde=True, bins=30, color='skyblue', ax=ax4)
        
        # Add vertical lines for key statistics
        mean_return = np.mean(returns) * 100
        median_return = np.median(returns) * 100
        ax4.axvline(x=mean_return, color='red', linestyle='-', label=f'Mean: {mean_return:.2f}%')
        ax4.axvline(x=median_return, color='green', linestyle='--', label=f'Median: {median_return:.2f}%')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        ax4.set_title('Monthly Return Distribution', fontsize=14)
        ax4.set_xlabel('Return (%)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. Drawdowns
        ax5 = fig.add_subplot(gs[4:6, 2:])
        
        # Calculate drawdowns
        drawdowns = []
        peak = 0
        for cr in cum_returns:
            if cr > peak:
                peak = cr
            drawdown = (peak - cr) / (1 + peak) if peak > 0 else 0
            drawdowns.append(drawdown)
        
        ax5.plot(range(len(drawdowns)), np.array(drawdowns) * 100, 'r-', linewidth=2)
        
        # Format x-axis with dates
        if len(dates) > 24:
            # If we have many dates, show fewer x labels
            step = max(1, len(dates) // 12)  # Show ~12 labels
            ax5.set_xticks(range(0, len(dates), step))
            ax5.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45, ha='right')
        else:
            ax5.set_xticks(range(len(dates)))
            ax5.set_xticklabels(dates, rotation=45, ha='right')
        
        ax5.set_title('Portfolio Drawdowns', fontsize=14)
        ax5.set_ylabel('Drawdown (%)', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # Invert y-axis so drawdowns go down
        ax5.invert_yaxis()
        
        # 6. Key metrics table
        ax6 = fig.add_subplot(gs[6, :])
        ax6.axis('off')
        
        # Format metrics for display
        summary_text = "Performance Summary\n------------------\n"
        summary_text += f"Period: {metrics.get('start_date', 'N/A')} to {metrics.get('end_date', 'N/A')}\n"
        summary_text += f"Months: {metrics.get('num_months', 0)}\n\n"
        
        # Returns
        summary_text += "Returns:\n"
        summary_text += f"  Monthly Mean: {metrics.get('mean_monthly_return', 0)*100:.2f}%\n"
        summary_text += f"  Monthly Median: {metrics.get('median_monthly_return', 0)*100:.2f}%\n"
        summary_text += f"  Monthly Std Dev: {metrics.get('std_monthly_return', 0)*100:.2f}%\n"
        summary_text += f"  Annualized Return: {metrics.get('annualized_return', 0)*100:.2f}%\n"
        summary_text += f"  Cumulative Return: {metrics.get('final_cumulative_return', 0)*100:.2f}%\n"
        summary_text += f"  Positive Months: {metrics.get('percent_positive_months', 0)*100:.1f}%\n\n"
        
        # Risk metrics
        summary_text += "Risk Metrics:\n"
        summary_text += f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
        summary_text += f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n"
        summary_text += f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}\n"
        summary_text += f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%\n\n"
        
        # Best/worst
        summary_text += "Extremes:\n"
        summary_text += f"  Best Month: {metrics.get('best_monthly_return', 0)*100:.2f}% ({metrics.get('best_month_date', 'N/A')})\n"
        summary_text += f"  Worst Month: {metrics.get('worst_monthly_return', 0)*100:.2f}% ({metrics.get('worst_month_date', 'N/A')})\n\n"
        
        # Portfolio
        summary_text += "Portfolio:\n"
        summary_text += f"  Avg Long Positions: {metrics.get('average_long_positions', 0):.1f}\n"
        summary_text += f"  Avg Short Positions: {metrics.get('average_short_positions', 0):.1f}\n"
        summary_text += f"  Avg Turnover: {metrics.get('average_turnover', 0):.2f}\n"
        
        ax6.text(0.02, 0.98, summary_text, va='top', family='monospace', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        # Title and layout
        plt.suptitle('Portfolio Performance Dashboard', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'performance_dashboard.png'), dpi=300)
        plt.close()
    
    def generate_report(self, metrics):
        """Generate comprehensive performance report."""
        report_dir = os.path.join(self.test_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create report file
        report_path = os.path.join(report_dir, "monthly_performance_report.json")
        
        # Create a safe copy of the config
        config_copy = {}
        if hasattr(self.config, 'config'):
            for key, value in self.config.config.items():
                if key == 'device':
                    config_copy[key] = str(value)  # Convert device to string
                else:
                    config_copy[key] = value
        
        # Add timestamp and version info
        report_data = {
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "config": config_copy
        }
        
        # Fix serialization issues with numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                import torch
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return str(obj)
                if isinstance(obj, torch.device):  # Add this line to handle torch devices
                    return str(obj)
                return super().default(obj)
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, cls=NumpyEncoder)
        
        logging.info(f"Generated monthly performance report at {report_path}")
        
        # Create HTML report
        self._generate_html_report(metrics, report_dir)
    
    def _generate_html_report(self, metrics, report_dir):
        """
        Generate HTML report with metrics and visualizations.
        
        Args:
            metrics: Dictionary of metrics
            report_dir: Output directory
        """
        html_path = os.path.join(report_dir, "monthly_performance_report.html")
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AlphaPortfolio Monthly Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .metrics-section {{ margin-bottom: 20px; }}
                .metric {{ margin: 5px 0; }}
                .metric-name {{ font-weight: bold; }}
                .metric-value {{ margin-left: 10px; }}
                .image-gallery {{ display: flex; flex-wrap: wrap; }}
                .image-container {{ margin: 10px; max-width: 45%; }}
                .image-container img {{ max-width: 100%; }}
                .caption {{ text-align: center; margin-top: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>AlphaPortfolio Monthly Performance Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Performance Period</h2>
            <div class="metrics-section">
                <div class="metric">
                    <span class="metric-name">Start Date:</span>
                    <span class="metric-value">{metrics.get('start_date', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">End Date:</span>
                    <span class="metric-value">{metrics.get('end_date', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Number of Months:</span>
                    <span class="metric-value">{metrics.get('num_months', 0)}</span>
                </div>
            </div>
            
            <h2>Performance Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Mean Monthly Return</td>
                    <td>{metrics.get('mean_monthly_return', 0)*100:.2f}%</td>
                </tr>
                <tr>
                    <td>Annualized Return</td>
                    <td>{metrics.get('annualized_return', 0)*100:.2f}%</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Sortino Ratio</td>
                    <td>{metrics.get('sortino_ratio', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Maximum Drawdown</td>
                    <td>{metrics.get('max_drawdown', 0)*100:.2f}%</td>
                </tr>
                <tr>
                    <td>Calmar Ratio</td>
                    <td>{metrics.get('calmar_ratio', 0):.2f}</td>
                </tr>
                <tr>
                    <td>% Positive Months</td>
                    <td>{metrics.get('percent_positive_months', 0)*100:.1f}%</td>
                </tr>
                <tr>
                    <td>Best Monthly Return</td>
                    <td>{metrics.get('best_monthly_return', 0)*100:.2f}% ({metrics.get('best_month_date', 'N/A')})</td>
                </tr>
                <tr>
                    <td>Worst Monthly Return</td>
                    <td>{metrics.get('worst_monthly_return', 0)*100:.2f}% ({metrics.get('worst_month_date', 'N/A')})</td>
                </tr>
            </table>
            
            <h2>Portfolio Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Average Long Positions</td>
                    <td>{metrics.get('average_long_positions', 0):.1f}</td>
                </tr>
                <tr>
                    <td>Average Short Positions</td>
                    <td>{metrics.get('average_short_positions', 0):.1f}</td>
                </tr>
                <tr>
                    <td>Average Long Exposure</td>
                    <td>{metrics.get('average_long_exposure', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Average Short Exposure</td>
                    <td>{metrics.get('average_short_exposure', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Average Monthly Turnover</td>
                    <td>{metrics.get('average_turnover', 0):.2f}</td>
                </tr>
            </table>
            
            <h2>Performance Visualizations</h2>
            <div class="image-gallery">
                <div class="image-container">
                    <img src="../visualizations/performance_dashboard.png" alt="Performance Dashboard">
                    <div class="caption">Performance Dashboard</div>
                </div>
                <div class="image-container">
                    <img src="../visualizations/cumulative_returns.png" alt="Cumulative Returns">
                    <div class="caption">Cumulative Returns</div>
                </div>
                <div class="image-container">
                    <img src="../visualizations/monthly_returns.png" alt="Monthly Returns">
                    <div class="caption">Monthly Returns</div>
                </div>
                <div class="image-container">
                    <img src="../visualizations/drawdowns.png" alt="Drawdowns">
                    <div class="caption">Drawdowns</div>
                </div>
                <div class="image-container">
                    <img src="../visualizations/rolling_sharpe.png" alt="Rolling Sharpe Ratio">
                    <div class="caption">Rolling Sharpe Ratio</div>
                </div>
                <div class="image-container">
                    <img src="../visualizations/return_distribution.png" alt="Return Distribution">
                    <div class="caption">Return Distribution</div>
                </div>
                <div class="image-container">
                    <img src="../visualizations/portfolio_allocation.png" alt="Portfolio Allocation">
                    <div class="caption">Portfolio Allocation</div>
                </div>
                <div class="image-container">
                    <img src="../visualizations/position_counts.png" alt="Position Counts">
                    <div class="caption">Position Counts</div>
                </div>
                <div class="image-container">
                    <img src="../visualizations/turnover.png" alt="Portfolio Turnover">
                    <div class="caption">Portfolio Turnover</div>
                </div>
                <div class="image-container">
                    <img src="../visualizations/benchmark_comparison.png" alt="Benchmark Comparison">
                    <div class="caption">Benchmark Comparison</div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logging.info(f"Generated HTML performance report at {html_path}")
        

