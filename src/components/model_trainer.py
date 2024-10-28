from src.exception.exception import CustomeException
from src.logging.logger import logging
import os,sys
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.config_entity import ModelTrainerConfig

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from src.utils.main_utlis.utils import evaluate_models,save_object,get_regression_score,load_object,load_numpy_array_data
from src.entity.artifact_entity import ModelTrainerArtifact

import mlflow
import json

import dagshub
dagshub.init(repo_owner='Tafique-Hossain-khan', repo_name='Food-Delivery-Time-Prediction', mlflow=True)

class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
                raise CustomeException(e,sys)
        
    def track_experiment(self, model, regression_metric,model_report_file_path,best_params):
        with mlflow.start_run():
            # Log regression metrics
            mlflow.log_metric("mean_absolute_error", regression_metric.mean_absolute_error)
            mlflow.log_metric("mean_squared_error", regression_metric.mean_squared_error)
            mlflow.log_metric("root_mean_squared_error", regression_metric.root_mean_squared_error)
            mlflow.log_metric("r2_score", regression_metric.r2_score)

            # Log the model
            mlflow.sklearn.log_model(model, "model")

            mlflow.log_artifact(model_report_file_path)
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            
        
    def train_model(self,X_train,y_train,x_test,y_test):
            try:
        

                models = {
                    "Random Forest Regressor": RandomForestRegressor(verbose=1)
                    
                }

                params = {
                    "Random Forest Regressor": {
                        'n_estimators': [50, 100, 200, 300],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                }

                model_report,best_params=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                                models=models,param=params)
                
                report_file_path = "model_report.json"
                with open(report_file_path, "w") as f:
                    json.dump(model_report, f)

                best_model_score = max(sorted(model_report.values()))

                best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]
                best_model = models[best_model_name]
                y_train_pred=best_model.predict(X_train)

                classification_train_metric=get_regression_score(y_true=y_train,y_pred=y_train_pred)
                
                ## Track the experiements with mlflow
                self.track_experiment(best_model,classification_train_metric,report_file_path,best_params)


                y_test_pred=best_model.predict(x_test)
                classification_test_metric=get_regression_score(y_true=y_test,y_pred=y_test_pred)

                self.track_experiment(best_model,classification_test_metric,report_file_path,best_params)

                preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
                    
                model_dir_path = os.path.dirname(self.model_trainer_config.final_model_file_path)
                os.makedirs(model_dir_path,exist_ok=True)

                save_object(self.model_trainer_config.final_model_file_path,obj=best_model)
                #model pusher
                save_object("final_model/model.pkl",best_model)
                

                ## Model Trainer Artifact
                model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.final_model_file_path,
                                    )
                logging.info(f"Model trainer artifact: {model_trainer_artifact}")
                return model_trainer_artifact
            except Exception as e:
                raise CustomeException(e,sys)


            


        
        
    
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise CustomeException(e,sys)
        
    