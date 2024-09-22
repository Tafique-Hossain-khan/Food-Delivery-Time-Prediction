from src.logger import logging
from src.exception import CustomException
from src.ingest_data import DataIngestorFactory
from zenml import step
import pandas as pd
import sys

@step
def data_injestion_step(file_path: str) -> pd.DataFrame:
    """ 
    Ingest the data from the csv file.
    
    Parameters:
    file_path : File path for the dataset

    Returns:
    A pandas DataFrame
    """
    try:
        file_extension = ".csv"
        csv_ingester = DataIngestorFactory.get_data_ingestor(file_extension)
        df = csv_ingester.ingest(file_path)
        logging.info(type(df))
        return df 
    
    except Exception as e:
        raise CustomException(e, sys)

    
'''
if __name__=="__main__":
    file_path = 'data/train.csv'
    data_injestion_step(file_path)
'''