from src.logging.logger import logging
from src.exception.exception import CustomeException
import pandas as pd
import numpy as np
import sys,os
from src.entity.artifact_entity import MissingValueImputerArtifact,DataIngestionArtifact
from src.entity.config_entity import MissingValueImputerConfig

class MissingValueImputer:
    def __init__(self, missing_value_imputer_config: MissingValueImputerConfig, data_ingestion_artifact:DataIngestionArtifact,
                strategy: str, column_strategies: dict = None):
        """
        This class aims to handle missing values in the dataframe

        Parameters:
        strategy: str 
            'Drop' or 'Fill'
        column_strategies: dict, optional
            A dictionary specifying the imputation strategy for specific columns
            e.g., {'column_name': ('method', fill_value)}
            where method can be 'mean', 'median', 'mode', or 'constant'
        """
        self.strategy = strategy
        self.missing_value_imputer_config = missing_value_imputer_config
        self.column_strategies = column_strategies or {}
        self.data_ingestion_artifact = data_ingestion_artifact

    def _get_auto_imputation_strategy(self, column: pd.Series) -> tuple:
        """Automatically determine the imputation strategy based on column characteristics"""
        if pd.api.types.is_numeric_dtype(column):
            if column.nunique() / len(column) < 0.05:  # If less than 5% unique values, use mode
                return ('mode', None)
            elif column.skew() > 1 or column.skew() < -1:  # If skewed, use median
                return ('median', None)
            else:
                return ('mean', None)
        elif pd.api.types.is_categorical_dtype(column) or pd.api.types.is_object_dtype(column):
            return ('mode', None)
        else:
            return ('constant', 'Unknown')

    def fill_missing_value(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()

        try:
            for column in df_cleaned.columns:
                if column in self.column_strategies:
                    method, fill_value = self.column_strategies[column]
                else:
                    method, fill_value = self._get_auto_imputation_strategy(df_cleaned[column])

                if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                    if method == 'mean':
                        df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)
                    elif method == 'median':
                        df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
                    elif method == 'mode':
                        df_cleaned[column].fillna(df_cleaned[column].mode()[0], inplace=True)
                    elif method == 'constant':
                        df_cleaned[column].fillna(fill_value, inplace=True)
                else:
                    if method in ['mean', 'median', 'mode']:
                        df_cleaned[column].fillna(df_cleaned[column].mode()[0], inplace=True)
                    elif method == 'constant':
                        df_cleaned[column].fillna(fill_value, inplace=True)

                logging.info(f"Imputed missing values in column '{column}' using method: {method}")

            return df_cleaned

        except Exception as e:
            raise CustomeException(e, sys)

    def drop_missing_value(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            initial_rows = len(df)
            df_cleaned = df.dropna()
            rows_dropped = initial_rows - len(df_cleaned)
            logging.info(f"Dropped {rows_dropped} rows with missing values.")
            return df_cleaned
        except Exception as e:
            raise CustomeException(e, sys)

    def initiate_missing_value_imputer(self):
        try:
            df = pd.read_csv(self.data_ingestion_artifact.raw_file_path)
            df = df.replace('NaN', float(np.nan), regex=True)
            if self.strategy == "Drop":
                df_cleaned = self.drop_missing_value(df)
            elif self.strategy == "Fill":
                df_cleaned = self.fill_missing_value(df)
            else:
                raise ValueError(f"Invalid strategy: {self.strategy}. Use 'Drop' or 'Fill'.")
            
            # Save the cleaned DataFrame
            os.makedirs(os.path.dirname(self.missing_value_imputer_config.cleaned_data_path),exist_ok=True)
            df_cleaned.to_csv(self.missing_value_imputer_config.cleaned_data_path, index=False)
            logging.info(f"Cleaned data saved to {self.missing_value_imputer_config.cleaned_data_path}")
            
            clean_data_artifact = MissingValueImputerArtifact(
                self.missing_value_imputer_config.cleaned_data_path
            )
            return clean_data_artifact
        
        except Exception as e:
            raise CustomeException(e, sys)
