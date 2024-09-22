from src.logger import logging
from src.exception import CustomException
from src.model_building import (
                                ModelBuilder,
                                ModelBuildingStrategy,
                                LinearRegressionStrategy,
                                RandomForestRegressorStrategy
                                )
from zenml import step,ArtifactConfig
from sklearn.pipeline import Pipeline   
import pandas as pd
import sys,os,pickle
import mlflow
from typing import Tuple,Annotated,Any
from zenml import Model
from zenml.client import Client

'''

experiment_tracker = Client().active_stack.experiment_tracker #is used to access the experiment tracking component of the active stack in ZenML

model = Model(
    name="Deliver_Time_prediction",
    version=None,
    description="Predict the delivery time"
)

'''
@step() 
def model_building_steps(X_train: pd.DataFrame, y_train: pd.Series,model_name:str):

    f"""
        Builds and trains a {model_name} model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Linear Regression model.
    """

    try:
        # Enable autologging for scikit-learn to automatically capture model metrics, parameters, and artifacts

        if model_name == "LinearRegression":
            model_strategy = ModelBuilder(LinearRegressionStrategy())
            
        elif model_name == "RandomForestRegressor":
            model_strategy = ModelBuilder(RandomForestRegressorStrategy())
        
        
        model = model_strategy.build_model(X_train,y_train)
        model_file_path = os.path.join("artifacts","model.pkl")
        with open(model_file_path,'wb') as f:
            pickle.dump(model,f)
        
        return model
    except Exception as e:
        raise CustomException(e,sys)
    
    

    