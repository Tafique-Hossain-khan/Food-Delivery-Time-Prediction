from abc import ABC ,abstractmethod
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import pandas as pd
from src.logger import logging

class FeatureEngineeringStrategy(ABC):
    
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """Abstract method to fit the transformation."""
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to apply transformation."""
        pass


class StandarScalerStrategy(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features = features
        self.scaler = StandardScaler()
        self.is_fitted = False  # To check if the scaler has been fitted

    def fit(self, df: pd.DataFrame):
        """
        Fits the scaler to the data.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the features to scale.
        """
        logging.info(f"Fitting standard scaler to features: {self.features}")
        self.scaler.fit(df[self.features])
        self.is_fitted = True
        logging.info("Standard scaler fitting completed.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing features to transform.

        Returns:
        pd.DataFrame: The DataFrame with scaled features.
        """
        if not self.is_fitted:
            raise RuntimeError("The scaler has not been fitted yet. Call 'fit' before 'transform'.")

        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed




class OneHotEncodingStrategy:
    def __init__(self, features):
        """
        Initializes the OneHotEncoder for converting the categorical features into numerical features.

        Parameters:
        features (list): The list of features to apply one-hot encoding to.
        """
        self.features = features
        self.encoder = OneHotEncoder()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Fits the encoder to the data.

        Parameter:
        df (pd.DataFrame): DataFrame containing the data to fit.

        Return:
        self: The fitted instance for chaining.
        """
        ohe = self.encoder.fit(df[self.features])
        self.is_fitted = True
        return ohe

    def transform(self, df: pd.DataFrame,instance) -> pd.DataFrame:
        """
        Transforms the data using the fitted encoder.

        Parameter:
        df (pd.DataFrame): DataFrame containing the data to transform.

        Return:
        pd.DataFrame: The transformed DataFrame with one-hot encoded features.
        """
        if not self.is_fitted:
            raise RuntimeError("The encoder has not been fitted yet. Call 'fit' before 'transform'.")

        encoded_df = pd.DataFrame(
            instance.transform(df[self.features]).toarray(),
            columns=instance.get_feature_names_out(self.features)
        ).reset_index(drop=True)

        df_transformed = df.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        return df_transformed
    

class FeatureEngineering:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        if not isinstance(strategy, FeatureEngineeringStrategy):
            raise TypeError("Expected strategy to be an instance of FeatureEngineeringStrategy.")
        self._strategy = strategy

    def fit(self, df: pd.DataFrame):
        #self.strategy.fit(df)
        return self.strategy.fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.transform(df)