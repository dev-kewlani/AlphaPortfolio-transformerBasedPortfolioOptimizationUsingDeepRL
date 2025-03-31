import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from data import AlphaPortfolioData

class SequentialAlphaPortfolioData(AlphaPortfolioData):
    """
    Extension of AlphaPortfolioData that enables sequential month-by-month evaluation.
    Provides a method to extract sequential data for monthly rebalancing tests.
    """
    
    def get_sequential_data(self):
        """
        Extract data in sequential order for monthly rebalancing tests.
        
        Returns:
            Dictionary with sequential data keyed by date
        """
        logging.info("Extracting sequential data for monthly testing")
        
        try:
            # Get unique dates from the dataset
            if not hasattr(self, 'data') or self.data is None:
                logging.error("No data available to extract sequential data")
                return None
            
            # Ensure dates are in datetime format
            self.data['date'] = pd.to_datetime(self.data['date'])
            
            # Get unique dates in sorted order
            unique_dates = sorted(self.data['date'].unique())
            logging.info(f"Found {len(unique_dates)} unique dates in dataset")
            
            # Process each date to create state, future returns, and masks
            all_states = []
            all_future_returns = []
            all_masks = []
            all_dates = []
            
            # We need at least lookback+1 dates (lookback for state, 1 for future return)
            for i in range(len(unique_dates) - self.lookback):
                current_date = unique_dates[i + self.lookback - 1]  # Current date is the last date in the lookback window
                future_date = unique_dates[i + self.lookback]  # Future date for returns
                
                # Get lookback period dates
                lookback_dates = unique_dates[i:i + self.lookback]
                
                # Create state for each asset
                state = np.zeros((self.global_max_assets, self.lookback, self.get_feature_count()))
                future_returns = np.zeros(self.global_max_assets)
                masks = np.zeros(self.global_max_assets, dtype=bool)
                
                # For each asset, fill in state and future return
                for permno_idx, permno in enumerate(self.unique_permnos):
                    # Get data for this asset in the lookback period
                    asset_lookback_data = self.data[(self.data['permno'] == permno) & 
                                                   (self.data['date'].isin(lookback_dates))]
                    
                    # Get future return data
                    asset_future_data = self.data[(self.data['permno'] == permno) & 
                                                (self.data['date'] == future_date)]
                    
                    # Only use assets that have complete data
                    if len(asset_lookback_data) == self.lookback and len(asset_future_data) == 1:
                        # Sort lookback data by date
                        asset_lookback_data = asset_lookback_data.sort_values('date')
                        
                        # Extract features (excluding 'permno', 'date', 'rdq')
                        features = asset_lookback_data.drop(columns=['permno', 'date', 'rdq']).values
                        
                        # Fill in state
                        state[permno_idx] = features
                        
                        # Fill in future return
                        future_returns[permno_idx] = asset_future_data['ret'].values[0]
                        
                        # Mark as valid
                        masks[permno_idx] = True
                
                # Store data
                all_states.append(state)
                all_future_returns.append(future_returns)
                all_masks.append(masks)
                all_dates.append(future_date)  # Use future date as the date label
            
            logging.info(f"Successfully extracted sequential data for {len(all_dates)} time points")
            
            # Return dictionary with sequential data
            return {
                'dates': all_dates,
                'states': np.array(all_states, dtype=np.float32),
                'future_returns': np.array(all_future_returns, dtype=np.float32),
                'masks': np.array(all_masks, dtype=bool)
            }
            
        except Exception as e:
            logging.error(f"Error extracting sequential data: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None

def create_sequential_test_loader(config, cycle_params, data_path, scaler_path=None):
    """
    Create a sequential test loader for month-by-month evaluation.
    
    Args:
        config: Configuration object
        cycle_params: Parameters for current cycle
        data_path: Path to data file
        scaler_path: Optional path to scaler for normalization
        
    Returns:
        Sequential test data loader
    """
    # Extract parameters
    T = config.config["model"]["T"]
    lookback = config.config["model"]["lookback"]
    experiment_id = config.config["experiment_id"]
    batch_size = config.config["training"]["batch_size"]
    num_workers = config.config["training"]["num_workers"]
    test_start = cycle_params.get("test_start", "2015-01-01")
    test_end = cycle_params.get("test_end", "2020-12-31")
    
    # Create sequential test data
    test_data = SequentialAlphaPortfolioData(
        data_path=data_path,
        start_date=test_start,
        end_date=test_end,
        T=T,
        lookback=lookback,
        cycle_id=999,  # Special ID for test
        experiment_id=experiment_id,
        scaler_path=scaler_path
    )
    
    # Create test loader
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logging.info(f"Created sequential test loader for period {test_start} to {test_end}")
    
    return test_loader, test_data