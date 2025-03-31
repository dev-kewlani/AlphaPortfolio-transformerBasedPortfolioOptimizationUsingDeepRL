from datetime import datetime
import os
from matplotlib.gridspec import GridSpec
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import traceback
from data import profile_function
from enhanced_visualization import VisualizationManager
import matplotlib.pyplot as plt
import torch.optim as optim
import seaborn as sns

def calculate_portfolio_std(
    weights: torch.Tensor, 
    future_returns: torch.Tensor,
    masks: Optional[torch.Tensor] = None, 
    return_step_values: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Calculate portfolio standard deviation from weights and future asset returns.
    
    Args:
        weights: Portfolio weights [batch_size, T, num_assets]
        future_returns: Future asset returns [batch_size, T, num_assets]
        masks: Optional masks for valid assets [batch_size, T, num_assets]
        return_step_values: Whether to return individual step values
        
    Returns:
        Portfolio standard deviation [batch_size] or
        Tuple of (std_deviation, step_returns)
    """
    # Calculate returns for each time step
    step_returns = calculate_returns(weights, future_returns, masks)  # [batch_size, T]
    
    # Calculate standard deviation across time steps
    std_dev = torch.std(step_returns, dim=1)  # [batch_size]
    
    if return_step_values:
        return std_dev, step_returns
    else:
        return std_dev

def calculate_sharpe_for_episode(
    returns: torch.Tensor,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 12.0
) -> torch.Tensor:
    """
    Calculate Sharpe ratio for an episode.
    
    Args:
        returns: Time series of returns [time_steps]
        risk_free_rate: Annual risk-free rate
        annualization_factor: Factor to annualize returns (12 for monthly)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return torch.tensor(0.0, device=returns.device)
    
    # Convert annual risk-free rate to period rate
    rf_period = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_period
    
    # Calculate Sharpe ratio
    mean_excess = torch.mean(excess_returns)
    std_excess = torch.std(excess_returns) + 1e-8  # Add small constant to avoid division by zero
    
    # Annualize Sharpe ratio
    sharpe = (mean_excess / std_excess) * torch.sqrt(torch.tensor(annualization_factor))
    
    return sharpe

def calculate_batch_sharpe(
    returns: torch.Tensor,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 12.0
) -> torch.Tensor:
    """
    Calculate Sharpe ratio for a batch of episodes.
    
    Args:
        returns: Time series of returns [batch_size, time_steps]
        risk_free_rate: Annual risk-free rate
        annualization_factor: Factor to annualize returns (12 for monthly)
        
    Returns:
        Sharpe ratios [batch_size]
    """
    batch_size, time_steps = returns.shape
    
    if time_steps < 2:
        return torch.zeros(batch_size, device=returns.device)
    
    # Convert annual risk-free rate to period rate
    rf_period = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_period
    
    # Calculate mean and std for each episode in the batch
    mean_excess = torch.mean(excess_returns, dim=1)  # [batch_size]
    std_excess = torch.std(excess_returns, dim=1) + 1e-8  # [batch_size]
    
    # Annualize Sharpe ratio
    sharpe = (mean_excess / std_excess) * torch.sqrt(torch.tensor(annualization_factor))
    
    return sharpe

def calculate_rolling_sharpe(
    returns: torch.Tensor,
    window_size: int = 12,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 12.0
) -> torch.Tensor:
    """
    Calculate rolling Sharpe ratio with a window.
    
    Args:
        returns: Time series of returns [batch_size, time_steps]
        window_size: Window size for rolling calculation
        risk_free_rate: Annual risk-free rate
        annualization_factor: Factor to annualize returns (12 for monthly)
        
    Returns:
        Rolling Sharpe ratios [batch_size, time_steps-window_size+1]
    """
    batch_size, time_steps = returns.shape
    
    if time_steps < window_size:
        return torch.zeros((batch_size, 1), device=returns.device)
    
    # Convert annual risk-free rate to period rate
    rf_period = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    
    # Calculate rolling Sharpe ratios
    rolling_sharpes = []
    
    for t in range(window_size, time_steps + 1):
        # Get window of returns
        window_returns = returns[:, t-window_size:t]  # [batch_size, window_size]
        
        # Calculate excess returns
        excess_returns = window_returns - rf_period
        
        # Calculate mean and std for this window
        mean_excess = torch.mean(excess_returns, dim=1)  # [batch_size]
        std_excess = torch.std(excess_returns, dim=1) + 1e-8  # [batch_size]
        
        # Calculate Sharpe
        sharpe = (mean_excess / std_excess) * torch.sqrt(torch.tensor(annualization_factor))
        
        rolling_sharpes.append(sharpe)
    
    # Stack all windows
    if rolling_sharpes:
        rolling_sharpes = torch.stack(rolling_sharpes, dim=1)  # [batch_size, time_steps-window_size+1]
    else:
        rolling_sharpes = torch.zeros((batch_size, 1), device=returns.device)
    
    return rolling_sharpes

def calculate_portfolio_stats(
    weights: torch.Tensor,
    future_returns: torch.Tensor,
    masks: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Calculate comprehensive portfolio statistics.
    
    Args:
        weights: Portfolio weights [batch_size, T, num_assets]
        future_returns: Future asset returns [batch_size, T, num_assets]
        masks: Optional masks for valid assets [batch_size, T, num_assets]
        
    Returns:
        Dictionary with portfolio statistics
    """
    # Calculate returns
    returns = calculate_returns(weights, future_returns, masks)  # [batch_size, T]
    
    # Calculate various statistics
    mean_returns = torch.mean(returns, dim=1)  # [batch_size]
    std_returns = torch.std(returns, dim=1) + 1e-8  # [batch_size]
    sharpe_ratios = mean_returns / std_returns * torch.sqrt(torch.tensor(12.0))  # [batch_size]
    
    # Calculate cumulative returns
    cum_returns = torch.cumprod(1 + returns, dim=1) - 1  # [batch_size, T]
    final_cum_returns = cum_returns[:, -1]  # [batch_size]
    
    # Count positive returns
    positive_returns = torch.mean((returns > 0).float(), dim=1)  # [batch_size]
    
    # Calculate max drawdown
    peak_values = torch.cummax(cum_returns, dim=1)[0]  # [batch_size, T]
    drawdowns = (peak_values - cum_returns) / (peak_values + 1e-8)  # [batch_size, T]
    max_drawdowns = torch.max(drawdowns, dim=1)[0]  # [batch_size]
    
    # Calculate long/short exposures
    long_exposure = torch.sum(torch.clamp(weights, min=0), dim=2)  # [batch_size, T]
    short_exposure = torch.sum(torch.clamp(weights, max=0), dim=2)  # [batch_size, T]
    mean_long = torch.mean(long_exposure, dim=1)  # [batch_size]
    mean_short = torch.mean(-short_exposure, dim=1)  # [batch_size]
    
    # Calculate turnover (if T > 1)
    if weights.shape[1] > 1:
        weight_changes = torch.diff(weights, dim=1)  # [batch_size, T-1, num_assets]
        turnover = torch.sum(torch.abs(weight_changes), dim=(1, 2)) / (weights.shape[1] - 1)  # [batch_size]
    else:
        turnover = torch.zeros_like(mean_returns)  # [batch_size]
    
    # Return all statistics
    return {
        'returns': returns,  # [batch_size, T]
        'mean_returns': mean_returns,  # [batch_size]
        'std_returns': std_returns,  # [batch_size]
        'sharpe_ratios': sharpe_ratios,  # [batch_size]
        'cum_returns': cum_returns,  # [batch_size, T]
        'final_cum_returns': final_cum_returns,  # [batch_size]
        'positive_returns': positive_returns,  # [batch_size]
        'max_drawdowns': max_drawdowns,  # [batch_size]
        'mean_long_exposure': mean_long,  # [batch_size]
        'mean_short_exposure': mean_short,  # [batch_size]
        'turnover': turnover  # [batch_size]
    }

def transfer_batch_to_device(batch, device):
    """Efficiently transfer a batch of data to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return [transfer_batch_to_device(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {key: transfer_batch_to_device(value, device) for key, value in batch.items()}
    else:
        return batch

# -------------------------
# Utility Functions
# -------------------------
def plot_std_distribution(std_values: np.ndarray, output_dir: str, 
                        filename: str = "std_distribution.png", 
                        title: str = "Portfolio Standard Deviation Distribution"):
    """
    Plot distribution of portfolio standard deviations.
    
    Args:
        std_values: Standard deviation values
        output_dir: Output directory
        filename: Output filename
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with KDE
    sns.histplot(std_values * 100, kde=True, bins=30)
    
    # Add mean line
    mean_std = np.mean(std_values) * 100
    plt.axvline(x=mean_std, color='red', linestyle='--', 
               label=f'Mean: {mean_std:.2f}%')
    
    # Add annotations
    plt.title(title, fontsize=14)
    plt.xlabel('Portfolio Standard Deviation (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    stats_text = (
        f"Mean: {mean_std:.2f}%\n"
        f"Median: {np.median(std_values) * 100:.2f}%\n"
        f"Min: {np.min(std_values) * 100:.2f}%\n"
        f"Max: {np.max(std_values) * 100:.2f}%"
    )
    plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sharpe_distribution(sharpe_values: np.ndarray, output_dir: str, 
                           filename: str = "sharpe_distribution.png", 
                           title: str = "Sharpe Ratio Distribution"):
    """
    Plot distribution of Sharpe ratios.
    
    Args:
        sharpe_values: Sharpe ratio values
        output_dir: Output directory
        filename: Output filename
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with KDE
    sns.histplot(sharpe_values, kde=True, bins=30)
    
    # Add mean line
    mean_sharpe = np.mean(sharpe_values)
    plt.axvline(x=mean_sharpe, color='red', linestyle='--', 
               label=f'Mean: {mean_sharpe:.4f}')
    
    # Add vertical line at 0
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Add annotations
    plt.title(title, fontsize=14)
    plt.xlabel('Sharpe Ratio', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    stats_text = (
        f"Mean: {mean_sharpe:.4f}\n"
        f"Median: {np.median(sharpe_values):.4f}\n"
        f"Min: {np.min(sharpe_values):.4f}\n"
        f"Max: {np.max(sharpe_values):.4f}\n"
        f"Positive Sharpe: {np.mean(sharpe_values > 0) * 100:.1f}%"
    )
    plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sharpe_vs_std(sharpe_values: np.ndarray, std_values: np.ndarray, 
                     output_dir: str, filename: str = "sharpe_vs_std.png", 
                     title: str = "Sharpe Ratio vs Standard Deviation"):
    """
    Plot Sharpe ratio against standard deviation for risk-return analysis.
    
    Args:
        sharpe_values: Sharpe ratio values
        std_values: Standard deviation values
        output_dir: Output directory
        filename: Output filename
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with coloring by Sharpe
    sc = plt.scatter(std_values * 100, sharpe_values, c=sharpe_values, 
                   cmap='viridis', alpha=0.7, s=80)
    
    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Sharpe Ratio', fontsize=12)
    
    # Add mean lines
    plt.axvline(x=np.mean(std_values) * 100, color='r', linestyle='--', alpha=0.5, 
               label=f'Mean Std Dev: {np.mean(std_values)*100:.2f}%')
    plt.axhline(y=np.mean(sharpe_values), color='g', linestyle='--', alpha=0.5, 
               label=f'Mean Sharpe: {np.mean(sharpe_values):.4f}')
    
    # Add horizontal line at Sharpe = 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Portfolio Standard Deviation (%)', fontsize=12)
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Optimized function to calculate returns
def calculate_returns(weights: torch.Tensor, future_returns: torch.Tensor, 
                    masks: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate portfolio returns efficiently from weights and future asset returns.
    
    Args:
        weights: Portfolio weights [batch_size, T, num_assets]
        future_returns: Future asset returns [batch_size, T, num_assets]
        masks: Optional masks for valid assets [batch_size, T, num_assets]
        
    Returns:
        Portfolio returns [batch_size, T]
    """
    # Apply masks if provided (do this once)
    if masks is not None:
        masked_weights = weights * masks
    else:
        masked_weights = weights
    
    # Calculate portfolio returns in one operation
    # This is more efficient than looping through time steps
    portfolio_returns = torch.sum(masked_weights * future_returns, dim=-1)  # [batch_size, T]
    
    return portfolio_returns


class RLTrainer:
    """
    Reinforcement Learning trainer for AlphaPortfolio.
    Handles training, validation, tracking and visualization of metrics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        max_grad_norm: float = 0.5,
        sharpe_window: int = 12,
        gamma: float = 0.99,
        device: torch.device = None
    ):
        """
        Initialize RL trainer.
        
        Args:
            model: AlphaPortfolio model
            lr: Learning rate
            weight_decay: Weight decay for regularization
            max_grad_norm: Maximum gradient norm for gradient clipping
            sharpe_window: Window size for Sharpe calculation
            gamma: Discount factor for RL
            device: Device to train on
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        self.sharpe_window = sharpe_window
        self.gamma = gamma
        
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                      "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        logging.info(f"RLTrainer initialized with device: {self.device}")
        
        # Initialize trackers
        self.batch_tracker = BatchTracker()
        self.episode_tracker = None  # Will be set later if needed
        self.vis_manager = None      # Will be set later if needed
        
        logging.info(f"Initialized RL trainer with lr={lr}, weight_decay={weight_decay}, device={device}")
    
    @profile_function
    def train_epoch(self, train_loader, epoch: int, output_dir: str, cycle_idx: int, 
        param_set_id: Optional[str] = None) -> Dict[str, float]:
        """Train for one epoch with parameter set specific directories."""
        self.model.train()
        metrics_list = []
        
        # Reset batch tracker at the beginning of each epoch
        self.batch_tracker.reset()
        
        # Create parameter-specific output directory
        if param_set_id:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
        else:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
        
        os.makedirs(epoch_dir, exist_ok=True)

        epoch_returns = []
        epoch_weights = []
        epoch_scores = []
        epoch_indices = []
        epoch_stds = []
        epoch_sharpes = []
        
        # Reset episode-level trackers if we have one
        if hasattr(self, 'episode_tracker') and self.episode_tracker is not None:
            self.episode_tracker.reset()
        
        # Initialize scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        try:
            logging.info(f"Starting training for epoch {epoch} with {len(train_loader)} batches")
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
                # Safely unpack the batch - now with dates
                if len(batch) == 4:  # We have dates
                    states, future_returns, masks, episode_dates = batch
                else:  # No dates
                    states, future_returns, masks = batch
                    episode_dates = None
                
                # Move data to device (except dates)
                states = states.to(self.device)
                future_returns = future_returns.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass with mixed precision
                self.optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        portfolio_weights, winner_scores, sorted_indices = self.model(states, masks)
                        
                        # Calculate portfolio returns
                        batch_size, T, num_assets = future_returns.shape
                        portfolio_returns_timestep = calculate_returns(
                            portfolio_weights,
                            future_returns,
                            masks
                        )  # [batch_size, T]
                        
                        # Calculate portfolio standard deviation
                        portfolio_stds = torch.std(portfolio_returns_timestep, dim=1) + 1e-8  # [batch_size]
                        
                        # Calculate mean portfolio returns
                        mean_portfolio_returns = torch.mean(portfolio_returns_timestep, dim=1)  # [batch_size]
                        
                        # Calculate Sharpe ratio for each episode
                        sharpe_ratios = (mean_portfolio_returns / portfolio_stds) * torch.sqrt(torch.tensor(12.0))
                        
                        # Use mean Sharpe ratio as reward
                        mean_sharpe = torch.mean(sharpe_ratios)
                        
                        # Loss is negative Sharpe ratio (we want to maximize Sharpe)
                        loss = -mean_sharpe
                    
                    # Backward pass with scaler
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # Standard forward pass without mixed precision
                    portfolio_weights, winner_scores, sorted_indices = self.model(states, masks)
                    
                    # Calculate portfolio returns
                    batch_size, T, num_assets = future_returns.shape
                    portfolio_returns_timestep = calculate_returns(
                        portfolio_weights,
                        future_returns,
                        masks
                    )  # [batch_size, T]
                    
                    # Calculate portfolio standard deviation
                    portfolio_stds = torch.std(portfolio_returns_timestep, dim=1) + 1e-8  # [batch_size]
                    
                    # Calculate mean portfolio returns
                    mean_portfolio_returns = torch.mean(portfolio_returns_timestep, dim=1)  # [batch_size]
                    
                    # Calculate Sharpe ratio for each episode
                    sharpe_ratios = (mean_portfolio_returns / portfolio_stds) * torch.sqrt(torch.tensor(12.0))
                    
                    # Use mean Sharpe ratio as reward
                    mean_sharpe = torch.mean(sharpe_ratios)
                    
                    # Loss is negative Sharpe ratio (we want to maximize Sharpe)
                    loss = -mean_sharpe
                    
                    # Regular backward pass
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Store batch data in the batch tracker
                try:
                    self.batch_tracker.add_batch(
                        batch_metrics={
                            'loss': loss.item(),
                            'sharpe_ratio': mean_sharpe.item(),
                            'mean_return': mean_portfolio_returns.mean().item(),
                            'std_return': portfolio_stds.mean().item()
                        },
                        returns=portfolio_returns_timestep.detach().cpu().numpy(),
                        weights=portfolio_weights.detach().cpu().numpy(),
                        sharpe_ratios=sharpe_ratios.detach().cpu().numpy()
                    )
                except Exception as e:
                    logging.error(f"Error adding batch data to tracker: {str(e)}")
                
                # Store episode metrics if we have episode tracker
                if hasattr(self, 'episode_tracker') and self.episode_tracker is not None:
                    for b in range(batch_size):
                        self.episode_tracker.add_episode(
                            returns=portfolio_returns_timestep[b].detach().cpu().numpy(),
                            weights=portfolio_weights[b].detach().cpu().numpy(),
                            winner_scores=winner_scores[b].detach().cpu().numpy() if winner_scores is not None else None,
                            portfolio_return=mean_portfolio_returns[b].item(),
                            portfolio_std=portfolio_stds[b].item(),
                            sharpe_ratio=sharpe_ratios[b].item()
                        )
                
                # Track for epoch-level metrics
                epoch_returns.extend(portfolio_returns_timestep.detach().cpu().numpy())
                epoch_stds.extend(portfolio_stds.detach().cpu().numpy())
                epoch_sharpes.extend(sharpe_ratios.detach().cpu().numpy())
                
                episode_dir = os.path.join(epoch_dir, "episodes")
                os.makedirs(episode_dir, exist_ok=True)
                
                # Visualize episodes 
                for ep_idx in range(batch_size):
                # Get dates for this episode if available
                    ep_dates = None
                    if episode_dates is not None:
                        ep_dates = episode_dates[ep_idx]
                    
                    self._visualize_detailed_portfolio(
                                    episode_idx=ep_idx,
                                    weights=portfolio_weights[ep_idx].detach().cpu().numpy(),
                                    returns=portfolio_returns_timestep[ep_idx].detach().cpu().numpy(),
                                    winner_scores=winner_scores[ep_idx].detach().cpu().numpy(),
                                    output_dir=episode_dir,
                                    epoch=epoch,
                                    batch_idx=batch_idx,
                                    phase="train",
                                    dates=ep_dates
                                )
                
                # Calculate batch metrics
                metrics = {
                    'loss': loss.item(),
                    'sharpe_ratio': mean_sharpe.item(),
                    'mean_return': mean_portfolio_returns.mean().item(),
                    'std_return': portfolio_stds.mean().item(),
                    'mean_weight_long': portfolio_weights[portfolio_weights > 0].mean().item() if (portfolio_weights > 0).any() else 0,
                    'mean_weight_short': portfolio_weights[portfolio_weights < 0].mean().item() if (portfolio_weights < 0).any() else 0
                }
                
                metrics_list.append(metrics)
                
                # Visualize batch 
                if batch_idx % 5 == 0 and hasattr(self, 'vis_manager') and self.vis_manager is not None:
                    self.vis_manager.visualize_batch(
                        batch_idx=batch_idx,
                        epoch=epoch,
                        phase='train',
                        returns=portfolio_returns_timestep.detach().cpu().numpy(),
                        weights=portfolio_weights.detach().cpu().numpy(),
                        sharpe_ratios=sharpe_ratios.detach().cpu().numpy(),
                        winner_scores=winner_scores.detach().cpu().numpy()
                    )
                
                # Log batch progress sparingly to avoid slowdown
                if batch_idx % 5 == 0:
                    metrics_avg = {k: np.mean([m[k] for m in metrics_list[-5:]]) for k in metrics_list[-1].keys()}
                    logging.info(f"  Batch {batch_idx}/{len(train_loader)}: {metrics_avg}")
            
            # Calculate average metrics
            metrics_avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys() if metrics_list}
            logging.info(f"Epoch {epoch} training complete: {metrics_avg}")
            
            return metrics_avg
                
        except Exception as e:
            logging.error(f"Error in train_epoch: {str(e)}")
            logging.error(f"Error traceback: {traceback.format_exc()}")
            # Return empty metrics in case of error
            return {'loss': float('inf'), 'sharpe_ratio': -float('inf'), 'mean_return': 0.0, 'std_return': 0.0}
    
    def _visualize_detailed_portfolio(self, episode_idx, weights, returns, winner_scores, output_dir, 
                               epoch=None, batch_idx=None, phase=None, dates=None):
        """
        Create detailed portfolio visualizations for a single episode.
        
        Args:
            episode_idx: Episode index or identifier
            weights: Portfolio weights for this episode
            returns: Returns for this episode
            winner_scores: Winner scores for this episode
            output_dir: Directory to save visualizations
            epoch: Optional epoch number for the filename
            batch_idx: Optional batch index for the filename
            phase: Optional phase name for the filename
            dates: Optional list of dates for this episode's time steps
        """
        # Create subfolder for this episode
        episode_folder_name = f"episode_{episode_idx}"
        
        # Add epoch and batch info to folder name if provided
        if epoch is not None and batch_idx is not None:
            episode_folder_name = f"epoch_{epoch}_batch_{batch_idx}_episode_{episode_idx}"
        elif epoch is not None:
            episode_folder_name = f"epoch_{epoch}_episode_{episode_idx}"
            
        if phase is not None:
            episode_folder_name = f"{phase}_{episode_folder_name}"
            
        episode_dir = os.path.join(output_dir, episode_folder_name)
        os.makedirs(episode_dir, exist_ok=True)
        
        # Title construction
        base_title = f"Episode {episode_idx}"
        if epoch is not None:
            base_title += f" (Epoch {epoch}"
            if batch_idx is not None:
                base_title += f", Batch {batch_idx}"
            base_title += ")"
        
        # Date range information for title if available
        date_range = ""
        if dates is not None and len(dates) > 0:
            try:
                start_date = pd.Timestamp(dates[0]).strftime('%Y-%m-%d')
                end_date = pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')
                date_range = f" [{start_date} to {end_date}]"
            except:
                pass
        
        # 1. Plot returns by time step
        self._plot_returns_by_timestep(
            returns=returns,
            output_dir=episode_dir,
            filename="returns.png",
            title=f"{base_title} Returns by Date{date_range}",
            dates=dates
        )
        
        # 2. Plot cumulative returns
        self._plot_cumulative_returns(
            returns=returns,
            output_dir=episode_dir,
            filename="cumulative_returns.png",
            title=f"{base_title} Cumulative Returns{date_range}",
            dates=dates
        )
        
        # 3. Plot weights at each time step
        for t in range(len(weights)):
            date_label = ""
            if dates is not None and t < len(dates):
                try:
                    date_label = pd.Timestamp(dates[t]).strftime('%Y-%m-%d')
                except:
                    date_label = f"t={t+1}"
            else:
                date_label = f"t={t+1}"
                
            self._plot_weights_at_timestep(
                weights=weights[t],
                output_dir=episode_dir,
                filename=f"weights_t{t+1}.png",
                title=f"{base_title} Portfolio Weights on {date_label}"
            )
        
        # 4. Plot portfolio allocation over time (heatmap)
        self._plot_portfolio_allocation_over_time(
            weights=weights,
            output_dir=episode_dir,
            filename="portfolio_allocation.png",
            title=f"{base_title} Portfolio Allocation Over Time{date_range}",
            dates=dates
        )
    
    def _visualize_aggregate_portfolio(self, all_returns, all_weights, output_dir):
        """Create aggregate portfolio visualizations across all episodes."""
        # 1. Plot distribution of returns
        plt.figure(figsize=(12, 8))
        sns.histplot(all_returns.flatten() * 100, kde=True, bins=30)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.axvline(x=np.mean(all_returns.flatten()) * 100, color='blue', linestyle='-', 
                label=f'Mean: {np.mean(all_returns.flatten()) * 100:.2f}%')
        
        plt.title('Distribution of Portfolio Returns', fontsize=14)
        plt.xlabel('Return (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        stats_text = (
            f"Mean: {np.mean(all_returns.flatten()) * 100:.2f}%\n"
            f"Median: {np.median(all_returns.flatten()) * 100:.2f}%\n"
            f"Std Dev: {np.std(all_returns.flatten()) * 100:.2f}%\n"
            f"Min: {np.min(all_returns.flatten()) * 100:.2f}%\n"
            f"Max: {np.max(all_returns.flatten()) * 100:.2f}%\n"
            f"Positive Returns: {np.mean(all_returns.flatten() > 0) * 100:.1f}%"
        )
        plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(output_dir, "returns_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot distribution of weights
        plt.figure(figsize=(12, 8))
        sns.histplot(all_weights.flatten(), kde=True, bins=30)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        plt.title('Distribution of Portfolio Weights', fontsize=14)
        plt.xlabel('Weight', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        stats_text = (
            f"Mean: {np.mean(all_weights.flatten()):.4f}\n"
            f"Median: {np.median(all_weights.flatten()):.4f}\n"
            f"Std Dev: {np.std(all_weights.flatten()):.4f}\n"
            f"Min: {np.min(all_weights.flatten()):.4f}\n"
            f"Max: {np.max(all_weights.flatten()):.4f}\n"
            f"Long Weights: {np.mean(all_weights.flatten() > 0) * 100:.1f}%\n"
            f"Short Weights: {np.mean(all_weights.flatten() < 0) * 100:.1f}%"
        )
        plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(output_dir, "weights_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Plot average cumulative returns across all episodes
        cum_returns = np.cumprod(1 + all_returns, axis=1) - 1
        mean_cum_returns = np.mean(cum_returns, axis=0)
        
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(mean_cum_returns) + 1), mean_cum_returns * 100, 'b-', linewidth=2)
        plt.fill_between(
            range(1, len(mean_cum_returns) + 1),
            (mean_cum_returns - np.std(cum_returns, axis=0)) * 100,
            (mean_cum_returns + np.std(cum_returns, axis=0)) * 100,
            alpha=0.2,
            color='blue'
        )
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.title('Average Cumulative Portfolio Returns', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "average_cumulative_returns.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_portfolio_distributions(self, all_returns, all_weights, output_dir):
        """Visualize distributions of portfolio metrics."""
        # Calculate Sharpe ratio for each episode
        episode_sharpes = []
        episode_returns = []
        episode_stds = []
        
        for i in range(all_returns.shape[0]):
            returns = all_returns[i]
            mean_return = np.mean(returns)
            std_return = np.std(returns) + 1e-8
            sharpe = mean_return / std_return * np.sqrt(12)  # Annualized
            
            episode_sharpes.append(sharpe)
            episode_returns.append(mean_return)
            episode_stds.append(std_return)
        
        # 1. Plot Sharpe ratio distribution
        plt.figure(figsize=(12, 8))
        sns.histplot(episode_sharpes, kde=True, bins=30)
        plt.axvline(x=np.mean(episode_sharpes), color='blue', linestyle='-', 
                label=f'Mean: {np.mean(episode_sharpes):.4f}')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        plt.title('Distribution of Episode Sharpe Ratios', fontsize=14)
        plt.xlabel('Sharpe Ratio', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        stats_text = (
            f"Mean: {np.mean(episode_sharpes):.4f}\n"
            f"Median: {np.median(episode_sharpes):.4f}\n"
            f"Std Dev: {np.std(episode_sharpes):.4f}\n"
            f"Min: {np.min(episode_sharpes):.4f}\n"
            f"Max: {np.max(episode_sharpes):.4f}\n"
            f"Positive Sharpe: {np.mean(np.array(episode_sharpes) > 0) * 100:.1f}%"
        )
        plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(output_dir, "sharpe_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot Sharpe vs Standard Deviation scatter
        plt.figure(figsize=(12, 8))
        plt.scatter(np.array(episode_stds) * 100, episode_sharpes, c=episode_sharpes, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Sharpe Ratio')
        
        plt.title('Sharpe Ratio vs. Standard Deviation', fontsize=14)
        plt.xlabel('Standard Deviation (%)', fontsize=12)
        plt.ylabel('Sharpe Ratio', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, "sharpe_vs_std.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Plot average portfolio positioning
        long_counts = np.mean(np.sum(all_weights > 0, axis=2), axis=0)
        short_counts = np.mean(np.sum(all_weights < 0, axis=2), axis=0)
        
        plt.figure(figsize=(12, 8))
        x = range(1, len(long_counts) + 1)
        
        plt.bar(x, long_counts, color='green', alpha=0.7, label='Long Positions')
        plt.bar(x, -short_counts, color='red', alpha=0.7, label='Short Positions')
        
        plt.title('Average Number of Positions by Time Step', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Number of Positions', fontsize=12)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, "average_positions.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_weights_at_timestep(self, weights: np.ndarray, output_dir: str, 
                                filename: str, title: str):
        """Plot portfolio weights at a specific time step."""
        plt.figure(figsize=(12, 10))
        
        # Sort weights by magnitude
        sorted_indices = np.argsort(-np.abs(weights))
        top_indices = sorted_indices[:min(20, len(weights))]
        
        # Get asset names
        if hasattr(self, 'assets_list') and len(self.assets_list) >= len(weights):
            top_assets = [self.assets_list[i] for i in top_indices]
        else:
            top_assets = [f"Asset {i}" for i in top_indices]
        
        # Create colors based on sign
        colors = ['green' if w > 0 else 'red' for w in weights[top_indices]]
        
        # Create horizontal bar chart
        plt.barh(top_assets, weights[top_indices], color=colors)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add annotations with exact weight values
        for i, w in enumerate(weights[top_indices]):
            plt.text(w + (0.001 if w >= 0 else -0.005), i, f'{w:.4f}', 
                    ha='left' if w >= 0 else 'right', va='center')
        
        plt.title(title, fontsize=14)
        plt.xlabel('Weight', fontsize=12)
        plt.ylabel('Asset', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        # Add summary statistics
        long_count = np.sum(weights > 0)
        short_count = np.sum(weights < 0)
        long_allocation = np.sum(weights[weights > 0])
        short_allocation = np.sum(weights[weights < 0])
        
        stats_text = (
            f"Long positions: {long_count}\n"
            f"Short positions: {short_count}\n"
            f"Long allocation: {long_allocation:.4f}\n"
            f"Short allocation: {short_allocation:.4f}"
        )
        
        plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_returns_by_timestep(self, returns: np.ndarray, output_dir: str, 
                            filename: str, title: str, dates=None):
        """Plot detailed returns at each time step with dates."""
        T = len(returns)
        plt.figure(figsize=(12, 6))
        
        # Format date labels
        if dates is not None and len(dates) == T:
            x_labels = []
            for date in dates:
                try:
                    if isinstance(date, (pd.Timestamp, datetime, np.datetime64)):
                        if hasattr(date, 'strftime'):
                            x_labels.append(date.strftime('%Y-%m-%d'))
                        else:
                            x_labels.append(pd.Timestamp(date).strftime('%Y-%m-%d'))
                    else:
                        x_labels.append(str(date))
                except Exception as e:
                    logging.warning(f"Error formatting date {date}: {str(e)}")
                    x_labels.append(f"t={len(x_labels)+1}")
        else:
            x_labels = [f"t={i+1}" for i in range(T)]
        
        # Create bar chart with colored bars based on sign
        colors = ['green' if r > 0 else 'red' for r in returns]
        plt.bar(range(T), returns * 100, color=colors)
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add return values as text
        for i, r in enumerate(returns):
            plt.text(i, r*100 + (0.5 if r >= 0 else -1.5), f'{r*100:.2f}%', 
                    ha='center', va='bottom' if r >= 0 else 'top')
        
        # Add mean return line
        mean_return = np.mean(returns) * 100
        plt.axhline(y=mean_return, color='blue', linestyle='-', 
                alpha=0.7, label=f'Mean: {mean_return:.2f}%')
        
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(range(T), x_labels, rotation=45, ha='right')
        plt.legend()
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cumulative_returns(self, returns: np.ndarray, output_dir: str, 
                           filename: str, title: str, dates=None):
        """Plot cumulative returns over time steps with dates."""
        if len(returns) == 0:
            return
            
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + returns) - 1
        
        plt.figure(figsize=(12, 6))
        
        # Format date labels
        if dates is not None and len(dates) == len(returns):
            x_labels = []
            for date in dates:
                try:
                    if isinstance(date, (pd.Timestamp, datetime, np.datetime64)):
                        if hasattr(date, 'strftime'):
                            x_labels.append(date.strftime('%Y-%m-%d'))
                        else:
                            x_labels.append(pd.Timestamp(date).strftime('%Y-%m-%d'))
                    else:
                        x_labels.append(str(date))
                except Exception as e:
                    logging.warning(f"Error formatting date {date}: {str(e)}")
                    x_labels.append(f"t={len(x_labels)+1}")
        else:
            x_labels = [f"t={i+1}" for i in range(len(returns))]
        
        # Plot cumulative returns
        plt.plot(range(len(returns)), cum_returns * 100, 'b-o', linewidth=2)
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add final return annotation
        final_return = cum_returns[-1] * 100
        plt.annotate(f"Final: {final_return:.2f}%", 
                    xy=(len(returns)-1, final_return), 
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center")
        
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(len(returns)), x_labels, rotation=45, ha='right')
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_portfolio_allocation_over_time(self, weights: np.ndarray, output_dir: str, 
                                      filename: str, title: str, dates=None):
        """Plot portfolio allocation over time with detailed visualizations."""
        if len(weights) == 0:
            return
        
        plt.figure(figsize=(14, 10))
        
        # Create a GridSpec for multiple subplots
        gs = GridSpec(2, 2, figure=plt.gcf())
        ax1 = plt.subplot(gs[0, :])  # Top panel for heatmap
        ax2 = plt.subplot(gs[1, 0])  # Bottom left for long positions
        ax3 = plt.subplot(gs[1, 1])  # Bottom right for short positions
        
        # Select top assets by absolute weight for visualization
        mean_abs_weights = np.mean(np.abs(weights), axis=0)
        top_indices = np.argsort(-mean_abs_weights)[:15]
        
        # Get asset names
        if hasattr(self, 'assets_list') and len(self.assets_list) >= weights.shape[1]:
            top_assets = [self.assets_list[i] for i in top_indices]
        else:
            top_assets = [f"Asset {i+1}" for i in top_indices]
        
        # Format date labels for x-axis
        if dates is not None and len(dates) == len(weights):
            # Format dates as strings
            date_labels = []
            for date in dates:
                try:
                    if isinstance(date, (pd.Timestamp, datetime, np.datetime64)):
                        if hasattr(date, 'strftime'):
                            date_labels.append(date.strftime('%Y-%m-%d'))
                        else:
                            date_labels.append(pd.Timestamp(date).strftime('%Y-%m-%d'))
                    else:
                        date_labels.append(str(date))
                except Exception as e:
                    logging.warning(f"Error formatting date {date}: {str(e)}")
                    date_labels.append(f"t={len(date_labels)+1}")
        else:
            date_labels = [f"t={i+1}" for i in range(len(weights))]
        
        # Create heatmap in top panel
        selected_weights = weights[:, top_indices]
        try:
            im = sns.heatmap(
                selected_weights.T,  # Transpose to have assets as rows
                cmap='RdBu_r',
                center=0,
                vmin=-np.max(np.abs(selected_weights)),
                vmax=np.max(np.abs(selected_weights)),
                yticklabels=top_assets,
                xticklabels=date_labels,
                cbar_kws={'label': 'Portfolio Weight'},
                ax=ax1
            )
            
            # Rotate date labels for better readability
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            ax1.set_title('Portfolio Allocation Over Time (Top 15 Assets)', fontsize=12)
            ax1.set_xlabel('Date', fontsize=10)
            ax1.set_ylabel('Asset', fontsize=10)
            
            # Calculate position counts for each time step
            long_counts = np.sum(weights > 0, axis=1)
            short_counts = np.sum(weights < 0, axis=1)
            
            # Plot position counts over time
            ax2.bar(range(len(weights)), long_counts, color='green', alpha=0.7, label='Long')
            ax2.bar(range(len(weights)), -short_counts, color='red', alpha=0.7, label='Short')
            ax2.set_title('Number of Positions by Date', fontsize=12)
            ax2.set_xlabel('Date', fontsize=10)
            ax2.set_ylabel('Count', fontsize=10)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_xticks(range(len(weights)))
            ax2.set_xticklabels(date_labels, rotation=45, ha='right')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Calculate allocation for each time step
            long_allocation = np.sum(np.maximum(weights, 0), axis=1)
            short_allocation = np.sum(np.minimum(weights, 0), axis=1)
            
            # Plot allocation over time
            ax3.bar(range(len(weights)), long_allocation, color='green', alpha=0.7, label='Long')
            ax3.bar(range(len(weights)), -short_allocation, color='red', alpha=0.7, label='Short')
            ax3.set_title('Allocation by Date', fontsize=12)
            ax3.set_xlabel('Date', fontsize=10)
            ax3.set_ylabel('Allocation', fontsize=10)
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            ax3.set_xticks(range(len(weights)))
            ax3.set_xticklabels(date_labels, rotation=45, ha='right')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.suptitle(title, fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        except Exception as e:
            logging.error(f"Error creating portfolio allocation plot: {str(e)}")
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
class EpisodeTracker:
    """Tracks metrics for individual episodes during training."""
    
    def __init__(self, T: int, num_assets: int):
        """
        Initialize episode tracker.
        
        Args:
            T: Number of time steps per episode
            num_assets: Number of assets in the universe
        """
        self.T = T
        self.num_assets = num_assets
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.returns = []            # List of [T] arrays
        self.weights = []            # List of [T, num_assets] arrays
        self.winner_scores = []      # List of [T, num_assets] arrays
        self.portfolio_returns = []  # List of scalar values
        self.portfolio_stds = []     # List of scalar values
        self.sharpe_ratios = []      # List of scalar values
    
    def add_episode(self, 
                  returns: np.ndarray,
                  weights: np.ndarray,
                  winner_scores: Optional[np.ndarray] = None,
                  portfolio_return: Optional[float] = None,
                  portfolio_std: Optional[float] = None,
                  sharpe_ratio: Optional[float] = None):
        """
        Add metrics for an episode.
        
        Args:
            returns: Returns for each time step [T]
            weights: Portfolio weights [T, num_assets]
            winner_scores: Winner scores [T, num_assets]
            portfolio_return: Portfolio return for the episode
            portfolio_std: Portfolio standard deviation
            sharpe_ratio: Sharpe ratio for the episode
        """
        self.returns.append(returns)
        self.weights.append(weights)
        
        if winner_scores is not None:
            self.winner_scores.append(winner_scores)
        
        if portfolio_return is not None:
            self.portfolio_returns.append(portfolio_return)
        
        if portfolio_std is not None:
            self.portfolio_stds.append(portfolio_std)
        
        if sharpe_ratio is not None:
            self.sharpe_ratios.append(sharpe_ratio)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics for all tracked episodes.
        
        Returns:
            Dictionary with aggregated metrics
        """
        metrics = {}
        
        if self.returns:
            # Stack returns and weights
            returns_array = np.stack(self.returns)
            weights_array = np.stack(self.weights)
            
            # Calculate aggregated metrics
            metrics['mean_return'] = float(np.mean(returns_array))
            metrics['std_return'] = float(np.std(returns_array))
            metrics['mean_return_by_step'] = np.mean(returns_array, axis=0).tolist()
            metrics['std_return_by_step'] = np.std(returns_array, axis=0).tolist()
            
            # Calculate portfolio statistics if available
            if self.portfolio_returns:
                metrics['mean_portfolio_return'] = float(np.mean(self.portfolio_returns))
                metrics['std_portfolio_return'] = float(np.std(self.portfolio_returns))
            
            if self.portfolio_stds:
                metrics['mean_portfolio_std'] = float(np.mean(self.portfolio_stds))
            
            if self.sharpe_ratios:
                metrics['mean_sharpe_ratio'] = float(np.mean(self.sharpe_ratios))
                metrics['median_sharpe_ratio'] = float(np.median(self.sharpe_ratios))
                metrics['min_sharpe_ratio'] = float(np.min(self.sharpe_ratios))
                metrics['max_sharpe_ratio'] = float(np.max(self.sharpe_ratios))
            
            # Weight statistics
            metrics['mean_weight_long'] = float(np.mean(weights_array[weights_array > 0])) if np.any(weights_array > 0) else 0.0
            metrics['mean_weight_short'] = float(np.mean(weights_array[weights_array < 0])) if np.any(weights_array < 0) else 0.0
            metrics['mean_num_long_positions'] = float(np.mean(np.sum(weights_array > 0, axis=2)))
            metrics['mean_num_short_positions'] = float(np.mean(np.sum(weights_array < 0, axis=2)))
        
        return metrics

class BatchTracker:
    """Tracks metrics for batches of episodes during training."""
    
    def __init__(self):
        """Initialize batch tracker."""
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.batch_metrics = []        # List of dictionaries with batch metrics
        self.batch_returns = []        # List of [batch_size, T] arrays
        self.batch_weights = []        # List of [batch_size, T, num_assets] arrays
        self.batch_winner_scores = []  # List of [batch_size, T, num_assets] arrays
        self.batch_sharpe_ratios = []  # List of [batch_size] arrays
    
    def add_batch(self,
                batch_metrics: Dict[str, float],
                returns: Optional[np.ndarray] = None,
                weights: Optional[np.ndarray] = None,
                winner_scores: Optional[np.ndarray] = None,
                sharpe_ratios: Optional[np.ndarray] = None):
        """
        Add metrics for a batch.
        
        Args:
            batch_metrics: Dictionary with metrics for this batch
            returns: Returns [batch_size, T]
            weights: Portfolio weights [batch_size, T, num_assets]
            winner_scores: Winner scores [batch_size, T, num_assets]
            sharpe_ratios: Sharpe ratios [batch_size]
        """
        self.batch_metrics.append(batch_metrics)
        
        if returns is not None:
            self.batch_returns.append(returns)
        
        if weights is not None:
            self.batch_weights.append(weights)
        
        if winner_scores is not None:
            self.batch_winner_scores.append(winner_scores)
        
        if sharpe_ratios is not None:
            self.batch_sharpe_ratios.append(sharpe_ratios)
    
    def get_epoch_metrics(self) -> Dict[str, float]:
        """
        Get aggregated metrics for the entire epoch.
        
        Returns:
            Dictionary with aggregated metrics
        """
        epoch_metrics = {}
        
        if self.batch_metrics:
            # Calculate mean of each metric across all batches
            for key in self.batch_metrics[0].keys():
                values = [batch[key] for batch in self.batch_metrics if key in batch]
                if values:
                    epoch_metrics[key] = float(np.mean(values))
        
        return epoch_metrics
    
    def get_all_returns(self) -> np.ndarray:
        """
        Get all returns from all batches with safety checks.
        
        Returns:
            Array of all returns or empty array if no data
        """
        if not self.batch_returns:
            logging.warning("No batch returns data collected")
            return np.array([])
        
        try:
            # Check each batch for valid data before stacking
            valid_returns = [returns for returns in self.batch_returns if returns is not None and returns.size > 0]
            
            if not valid_returns:
                logging.warning("No valid return data found in batch tracker")
                return np.array([])
                
            # Concatenate valid batch returns
            return np.vstack(valid_returns)
        except Exception as e:
            logging.error(f"Error in get_all_returns: {str(e)}")
            return np.array([])
    
    def get_all_weights(self) -> np.ndarray:
        """
        Get all weights from all batches with safety checks.
        
        Returns:
            Array of all weights or empty array if no data
        """
        if not self.batch_weights:
            logging.warning("No batch weights data collected")
            return np.array([])
        
        try:
            # Check each batch for valid data before stacking
            valid_weights = [weights for weights in self.batch_weights if weights is not None and weights.size > 0]
            
            if not valid_weights:
                logging.warning("No valid weight data found in batch tracker")
                return np.array([])
                
            # Concatenate valid batch weights
            return np.vstack(valid_weights)
        except Exception as e:
            logging.error(f"Error in get_all_weights: {str(e)}")
            return np.array([])

    def get_all_sharpe_ratios(self) -> np.ndarray:
        """
        Get all Sharpe ratios from all batches with safety checks.
        
        Returns:
            Array of all Sharpe ratios or empty array if no data
        """
        if not self.batch_sharpe_ratios:
            logging.warning("No batch Sharpe ratios data collected")
            return np.array([])
        
        try:
            # Check each batch for valid data before concatenating
            valid_sharpes = [sharpe for sharpe in self.batch_sharpe_ratios if sharpe is not None and sharpe.size > 0]
            
            if not valid_sharpes:
                logging.warning("No valid Sharpe ratio data found in batch tracker")
                return np.array([])
                
            # Concatenate valid batch Sharpe ratios
            return np.concatenate(valid_sharpes)
        except Exception as e:
            logging.error(f"Error in get_all_sharpe_ratios: {str(e)}")
            return np.array([])

class TrainingManager:
    """
    Comprehensive training manager for AlphaPortfolio.
    Handles training, validation, visualization, and logging.
    """
    
    def __init__(self, 
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               config: Any,
               cycle_params: Dict[str, Any],
               device: torch.device,
               sharpe_window: int = 12,
               max_grad_norm: float = 0.5):
        """
        Initialize training manager.
        
        Args:
            model: AlphaPortfolio model
            optimizer: Optimizer
            config: Configuration object
            cycle_params: Parameters for current cycle
            device: Device to train on
            sharpe_window: Window size for Sharpe calculation
            max_grad_norm: Maximum gradient norm for gradient clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.cycle_params = cycle_params
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                      "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        logging.info(f"TrainingManager initialized with device: {self.device}")
        self.sharpe_window = sharpe_window
        self.max_grad_norm = max_grad_norm
        
        # Extract parameters
        self.cycle_idx = cycle_params["cycle_idx"]
        self.output_dir = config.config["paths"]["plot_dir"]
        self.model_dir = config.config["paths"]["model_dir"]
        self.T = config.config["model"]["T"]
        self.num_epochs = config.config["training"]["num_epochs"]
        self.early_stopping_patience = config.config["training"]["patience"]
        
        # Create visualization manager
        self.vis_manager = None  # Will be initialized later with assets_list
        
        # Initialize trackers
        self.episode_tracker = EpisodeTracker(T=self.T, num_assets=0)  # Will be updated with actual num_assets
        self.train_batch_tracker = BatchTracker()
        self.val_batch_tracker = BatchTracker()
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'train_sharpe': [],
            'train_returns': [],
            'train_std': [],
            'val_loss': [],
            'val_sharpe': [],
            'val_returns': [],
            'val_std': [],
            'mean_weight_long': [],
            'mean_weight_short': []
        }
        
        # Initialize early stopping
        self.best_val_sharpe = -float('inf')
        self.early_stopping_counter = 0
        self.early_stop = False
        
        logging.info(f"Initialized training manager for cycle {self.cycle_idx}")
    
    def initialize_visualization(self, num_assets: int, assets_list: Optional[List[str]] = None):
        
        """
        Initialize visualization manager.
        
        Args:
            num_assets: Number of assets in the universe
            assets_list: Optional list of asset names
        """
        self.episode_tracker = EpisodeTracker(T=self.T, num_assets=num_assets)
        self.vis_manager = VisualizationManager(
            output_dir=self.output_dir,
            cycle_idx=self.cycle_idx,
            T=self.T,
            num_assets=num_assets,
            assets_list=assets_list
        )
        
