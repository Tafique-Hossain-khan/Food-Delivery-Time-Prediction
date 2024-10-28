from src.logging.logger import logging
from src.exception.exception import CustomeException
from dataclasses import dataclass
import os,sys,pickle
import pandas as pd
import yaml
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from src.entity.artifact_entity import RegressionMetricArtifact


def read_yaml(yaml_file_path):
    try:
        with open(yaml_file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)

        return yaml_content
    except Exception as e:
        raise CustomeException(e,sys)
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as file:
            yaml.dump(content, file)
        
    except Exception as e:
        raise CustomeException(e, sys)

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomeException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise CustomeException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomeException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomeException(e, sys) from e
    


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report,gs.best_params_

    except Exception as e:
        raise CustomeException(e, sys)
    



def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    try:
        model_mae = mean_absolute_error(y_true, y_pred)
        model_mse = mean_squared_error(y_true, y_pred)
        model_rmse = sqrt(model_mse)  # Calculating RMSE from MSE
        model_r2_score = r2_score(y_true, y_pred)

        regression_metric = RegressionMetricArtifact(
            mean_absolute_error=model_mae,
            mean_squared_error=model_mse,
            root_mean_squared_error=model_rmse,
            r2_score=model_r2_score
        )
        
        return regression_metric
    except Exception as e:
        raise CustomeException(e, sys)
