from src.logger import logging
from src.exception import CustomException
from zenml import step
import pandas as pd
import numpy as np
import sys

from src.handeling_missing_value import MissingValueHandler,DropMissingValuesStrategy,FillMissingValuesStrategy

@step
def handle_missing_values_step(
    df: pd.DataFrame, 
    strategy: str = None, 
    threshold=None,  
    column_strategies:dict=None
) -> pd.DataFrame:
    """
    Handles missing values using MissingValueHandler and the specified strategy.

    Parameters:
    df : pandas dataframe
    strategy :  'drop', or 'fill'
    fill_value: Any other fixed value or a list for 'random' strategy
    threshold : Threshold to drop the row (for future enhancement)
    column_strategies : A dictionary mapping column names to strategies. Example: 
                        {'col1': ('mean', None), 'col2': ('constant', 0)}

    Returns:
    pd.DataFrame: Cleaned dataframe with missing values handled.
    """
    # Drop strategy for dropping missing values
    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValuesStrategy(axis=0,thresh=threshold))
    
    # If specific strategies for columns are provided
    elif strategy == "fill":
        # Ensure fill values are native Python types before passing to handler
        #for col, (method, fill_value) in column_strategies.items():
            #if isinstance(fill_value, list) and len(fill_value) == 1:
                        #fill_value = fill_value[0]

        logging.info(f"column strategies before passing to function {column_strategies}")
        handler = MissingValueHandler(FillMissingValuesStrategy(column_strategies=column_strategies))
    
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")

    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df


@step
def conver_nan_type(df:pd.DataFrame) -> pd.DataFrame:
    """
    The nan is in the form of string and can't be recoginized by the pandas function


    """
    df = df.replace('NaN', float(np.nan), regex=True)
    return df