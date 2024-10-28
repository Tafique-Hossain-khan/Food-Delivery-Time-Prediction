from src.exception.exception import CustomeException
from src.logging.logger import logging
from dataclasses import dataclass
import os
import sys
import numpy as np
import pandas as pd
#import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config_obj:DataIngestionConfig):
        self.data_ingestion_config_obj = data_ingestion_config_obj

    def get_data(self):
        """
        This function amis to get the data from mongdb

        Parameters:
        db_name:Name of the database
        collection_name : Name of the collection where the data is stored

        Return:
        pd.Dataframe
        """
        try:
            df = pd.read_csv("Data\delivary_time.csv")
            
            
            os.makedirs(os.path.dirname(self.data_ingestion_config_obj.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config_obj.raw_data_path,index=False)
            
            return df
        except Exception as e:
            raise CustomeException(e,sys)
        
    def split_data_into_train_test(self,raw_data,test_size):
        try:
            logging.info("spliting the data into train and test")
    
            train_arr, test_arr = train_test_split(raw_data, test_size=test_size, random_state=42)

            logging.info(train_arr.shape)
            logging.info(test_arr.shape)

            os.makedirs(os.path.dirname(self.data_ingestion_config_obj.train_data_path),exist_ok=True)
            train_arr.to_csv(self.data_ingestion_config_obj.train_data_path,index=False)

            os.makedirs(os.path.dirname(self.data_ingestion_config_obj.test_data_path),exist_ok=True)
            test_arr.to_csv(self.data_ingestion_config_obj.test_data_path,index=False)

            logging.info("Data spliting complited and saved to the respective location")

            return train_arr,test_arr
        except Exception as e:
            raise CustomeException(e,sys)

    def initiate_data_ingestion(self):
        try:
            
            dataframe=self.get_data()
            self.split_data_into_train_test(dataframe,test_size=0.2)
            dataingestionartifact=DataIngestionArtifact(raw_file_path = self.data_ingestion_config_obj.raw_data_path,
                                                        trained_file_path=self.data_ingestion_config_obj.train_data_path,
                                                        test_file_path=self.data_ingestion_config_obj.test_data_path)
            return dataingestionartifact

        except Exception as e:
            raise CustomeException(e,sys)

    