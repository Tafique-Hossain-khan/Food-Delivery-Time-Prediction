from zenml import Model, pipeline, step
import pandas as pd
import sys,os,pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from src.logger import logging
from src.exception import CustomException
from steps.data_ingestion_step import data_injestion_step
from steps.handleing_missing_value_step import handle_missing_values_step,conver_nan_type
from steps.feature_extraction_steps import feature_extraction_steps
from steps.feature_engineering_steps import feature_engineering_steps
from steps.data_splitter_steps import data_splitter_step
from steps.ohe import ohe_step # temp solution 
from steps.model_building_steps import model_building_steps
from steps.model_evaluation_step import model_evaluation_step
from src.helper import column_strategies
@pipeline()

def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    try:
        logging.info("Data ingestion step")

        file_path = 'data/train.csv'
        raw_data  = data_injestion_step(file_path)
        
        logging.info("Data ingestion completed")
        logging.info(type(raw_data))


        #raw_data.drop(['ID','Delivery_person_ID'],axis=1,inplace=True)

        raw_data = feature_extraction_steps(df=raw_data,type_of_extraction="values")

        # Handleing the missing value
        #------------------------------
        logging.info("Removing the nan value")
        # raw_data = raw_data.replace('NaN', float(np.nan), regex=True) -> Adding it to handleing_missing_value_step.py
        logging.info(type(raw_data))
        
        
        raw_data = conver_nan_type(df=raw_data)
        raw_data = handle_missing_values_step(df= raw_data,column_strategies=column_strategies,strategy="fill")
        

        logging.info("nan valued filled sucessfully")
        # Feature Extraction
        #---------------------------
        logging.info("Starting The Data Extraction Process")
        #extracing relevent values from exising column
        #raw_data = feature_extraction_steps(df=raw_data,type_of_extraction="values")
        #updating the datatype
        
        raw_data = feature_extraction_steps(df=raw_data,type_of_extraction="update_datatype")
        
        
        #logging.info(raw_data.isnull().sum().sort_values(ascending=False)) # nan value after removing


        # Feature Extraction 
        #------------------------
        
        # Extracting date feature
        raw_data = feature_extraction_steps(df=raw_data,type_of_extraction="dates")

        # Calculating time differnce
        raw_data = feature_extraction_steps(df=raw_data,type_of_extraction="calculate_time")
        
        #Calculate distance between restaurant location & delivery location
        raw_data = feature_extraction_steps(df=raw_data,type_of_extraction="calculate_distance")
        

        #raw_data.drop(columns=['ID', 'Delivery_person_ID'],axis=1,inplace=True)

        

        # Train test split
        #----------------------------
        X_train, X_test, y_train, y_test = data_splitter_step(df=raw_data,target_column='Time_taken(min)')
    
        
        # Feature Engineering
        #-------------------------

        ''' 
        # Fit the encoder on X_train and store the fitted pipeline
        ohe = feature_engineering_steps(
            df=X_train,
            strategy_type="OneHotEncoding",
            features_list=cat_col,
            transformation_type="fit"
        )

        logging.info(f"Fitted OneHotEncoder pipeline type: {type(pipeline)}")

        # Transform X_train using the fitted pipeline
        X_train_final = feature_engineering_steps(
            df=X_train,
            strategy_type="OneHotEncoding",
            features_list=cat_col,
            transformation_type="transform",
            instance=ohe  # Pass the fitted pipeline
        )

        logging.info(f"Transformed X_train with OneHotEncoder")

        # Transform X_test using the same fitted pipeline
        X_test_final = feature_engineering_steps(
            df=X_test,
            strategy_type="OneHotEncoding",
            features_list=cat_col,
            transformation_type="transform",
            instance=ohe  # Use the same fitted pipeline for X_test
        )

        logging.info(f"Transformed X_test with OneHotEncoder")


        logging.info(X_train_final.shape)
        logging.info(X_train_final.columns)
        '''
        # Temp soltion for encoding categorical variable
        #--------------------------------------------------
        # Fit and transform the categorical columns
        X_train_final,X_test_final,y_train, y_test = ohe_step(X_train, X_test, y_train, y_test)

        
        # Model training
        #-------------------
        
        model = model_building_steps(X_train_final,y_train,model_name = "RandomForestRegressor")

        
        
        # Evaluation pipeline
        #---------------------
        evaluation_metrics, mse =  model_evaluation_step(model,X_test_final,y_test)
        logging.info(evaluation_metrics)
        logging.info(mse)
        
        return model

    except Exception as e:
        raise CustomException(e,sys)


if __name__ == "__main__":
    ml_pipeline()