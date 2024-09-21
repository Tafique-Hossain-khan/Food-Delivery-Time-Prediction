import pandas as pd
from src.logger import logging
from src.exception import CustomException
import sys
from abc import ABC,abstractmethod

# Abstract Base Class for Missing Value Handling Strategy
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        pass

#meadn medina,mode,random value,drop

# Concrete Strategy for Dropping Missing Values
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        """
        Initializes the DropMissingValuesStrategy with specific parameters.

        Parameters:
        axis (int): 0 to drop rows with missing values, 1 to drop columns with missing values.
        thresh (int): The threshold for non-NA values. Rows/Columns with less than thresh non-NA values are dropped.
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis and threshold.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values dropped.
        """
        try:
            logging.info(f"Dropping missing values with axis={self.axis} and thresh={self.thresh}")
            df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
            logging.info("Missing values dropped.")
            return df_cleaned
        except Exception as e:
            raise CustomException(e,sys)
        

class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, column_strategies=None) -> None:
        self.column_strategies = column_strategies if column_strategies else {}

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()

        try:
            for column, (method, fill_value) in self.column_strategies.items():
                if column not in df.columns:
                    raise ValueError(f"Column '{column}' not found in DataFrame.")
                
                if method == "mean":
                    if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                        df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)
                    else:
                        logging.warning(f"Method 'mean' cannot be applied to non-numeric column '{column}'.")
                
                elif method == "median":
                    if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                        df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
                    else:
                        logging.warning(f"Method 'median' cannot be applied to non-numeric column '{column}'.")
                
                elif method == "mode":
                    df_cleaned[column].fillna(df_cleaned[column].mode()[0], inplace=True)

                elif method == "constant":
                    if isinstance(fill_value, list) and len(fill_value) == 1:
                        fill_value = fill_value[0]
                    df_cleaned[column].fillna(fill_value, inplace=True)

                else:
                    logging.warning(f"Unknown method '{method}' for column '{column}'. No missing values handled.")
            
            logging.info("Missing values filled.")
            return df_cleaned
        
        except Exception as e:
            raise CustomException(e, sys)

        
        
class MissingValueHandler:

    def __init__(self,strategy:MissingValueHandlingStrategy) -> None:
        """
        Initializes the MissingValueHandler with a specific missing value handling strategy.

        Parameters:
        strategy (MissingValueHandlingStrategy): The strategy to be used for handling missing values.
        """
        self._strategy = strategy
        pass

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """
        Sets a new strategy for the MissingValueHandler.

        Parameters:
        strategy (MissingValueHandlingStrategy): The new strategy to be used for handling missing values.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the missing value handling using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)