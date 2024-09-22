from src.feature_extraction import (FeatureExtractionHandler,
                                    ExtractColumnValuesStrategy,
                                    ExtractDateFeaturesStrategy,
                                    UpdateDatatypesStrategy,
                                    CalculateDistanceStrategy,
                                    CalculateTimeDiffStrategy
                                    )

from zenml import step
import pandas as pd


@step
def feature_extraction_steps(df:pd.DataFrame,type_of_extraction:str):
    """
    Extract the feature form the existing feature

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    strategy : Feature extracion strategy.
    type_of_extraction (str) : Define what to extract from the column (values,dates,calculate_time,update_datatype,calculate_distance)

    Returns:
    pd.DataFrame: The DataFrame with features extracted
    """

    if type_of_extraction=="values":
        transformed_handler = FeatureExtractionHandler(strategy=ExtractColumnValuesStrategy())

    elif type_of_extraction=="dates":
        transformed_handler = FeatureExtractionHandler(strategy=ExtractDateFeaturesStrategy())
        
    elif type_of_extraction=="calculate_time":
        transformed_handler = FeatureExtractionHandler(strategy=CalculateTimeDiffStrategy())
        
    elif type_of_extraction=="update_datatype":
        transformed_handler = FeatureExtractionHandler(strategy=UpdateDatatypesStrategy())

    elif type_of_extraction=="calculate_distance":
        transformed_handler = FeatureExtractionHandler(strategy=CalculateDistanceStrategy())
        
    
    transformed_df= transformed_handler.run(df=df)

    return transformed_df
