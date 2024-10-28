from src.exception.exception import CustomeException
from src.logging.logger import logging
import sys,os
from src.entity.artifact_entity import FeatureExtractionArtifact
from src.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
from src.entity.artifact_entity import DataTransformationArtifact

from src.utils.main_utlis.utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self, feature_extraction_artifact: FeatureExtractionArtifact, data_transformation_config: DataTransformationConfig):
        self.feature_extraction_artifact = feature_extraction_artifact
        self.data_transformation_config = data_transformation_config

    def get_transformation_obj(self):
        try:
            # Load the dataset
            df = pd.read_csv(self.feature_extraction_artifact.cleaned_data_file_path)

            # Identify numerical and categorical columns
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

            # Create preprocessing steps for numerical columns
            '''
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            '''
            # Create preprocessing steps for categorical columns
            
            cat_pipeline = Pipeline(
                steps=[
                    
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            

            # Combine all the preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    #('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ],
                remainder='passthrough'  # This will pass through any columns that were not specified
            )

            logging.info("Preprocessing object created successfully")
            return preprocessor

        except Exception as e:
            raise CustomeException(e, sys)

    def initiate_data_transformation(self):
        try:
            # Get the preprocessing object
            preprocessing_obj = self.get_transformation_obj()

            # Load the dataset
            df = pd.read_csv(self.feature_extraction_artifact.cleaned_data_file_path)

            # Assuming the target column is 'Time_taken(min)'
            target_column_name = 'Time_taken(min)'
            
            # Separate features and target
            X = df.drop(columns=[target_column_name], axis=1)
            y = df[target_column_name]

            # Perform train-test split
            input_feature_train, input_feature_test, target_train, target_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Fit on training data and transform both training and test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test)

            # Combine processed features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_train)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test)]

            # Save the transformed data
            transformed_train_file_path = os.path.join(self.data_transformation_config.transformed_train_file_path)
            transformed_test_file_path = os.path.join(self.data_transformation_config.transformed_test_file_path)
            
            save_numpy_array_data(transformed_train_file_path, array=train_arr)
            save_numpy_array_data(transformed_test_file_path, array=test_arr)

            # Save the preprocessing object
            preprocessing_obj_file_path = self.data_transformation_config.transformed_object_file_path

            save_object(preprocessing_obj_file_path , preprocessing_obj)

            save_object( "final_model/preprocessor.pkl", preprocessing_obj)


            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise CustomeException(e, sys)
