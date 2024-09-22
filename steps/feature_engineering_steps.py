from zenml import step
from src.feature_engineering import (FeatureEngineeringStrategy,
                                    FeatureEngineering,
                                    StandarScalerStrategy,
                                    OneHotEncodingStrategy)

import pandas as pd
import os,pickle

@step
def feature_engineering_steps(df: pd.DataFrame, strategy_type: str, features_list: list, transformation_type: str, instance=None):
    """
    Steps to apply feature engineering.

    Parameters:
    df (pd.DataFrame): Pandas DataFrame as input.
    strategy_type (str): Strategy for applying Feature Engineering (OneHotEncoding, StandardScaler).
    features_list (list): The list of features to apply the transformation to.
    transformation_type (str): "fit" or "transform".
    feature_eng_handler: Pre-fitted handler to be used for transformation (optional).
    """
    if transformation_type == "fit":
        if strategy_type == "OneHotEncoding":
            strategy = OneHotEncodingStrategy(features=features_list)
        else:
            raise ValueError("Not a valid strategy")
        
        ohe = strategy.fit(df)
        return ohe

    elif transformation_type == "transform":
        
        return strategy.transform(df,ohe)

    else:
        raise ValueError("Invalid transformation_type. Must be 'fit' or 'transform'.")
