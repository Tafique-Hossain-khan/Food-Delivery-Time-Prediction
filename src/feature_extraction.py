from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from src.logger import logging

class FeatureExtractionStrategy(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to perform feature extraction on the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame to transform.

        Returns:
        pd.DataFrame: The DataFrame with new features extracted.
        """
        pass
class ExtractColumnValuesStrategy(FeatureExtractionStrategy):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        logging.info("Feature Extraction started")
        logging.info(type(df))


        df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: int(x.split(' ')[1].strip()))
        df.rename(columns={'Weatherconditions': 'Weather_conditions'},inplace=True)
        df['Weather_conditions'] = df['Weather_conditions'].apply(lambda x: x.split(' ')[1].strip())
        #df['City_code'] = df['Delivery_person_ID'].str.split("RES", expand=True)[0]
        df['Road_traffic_density'] = df['Road_traffic_density'].str.strip()
        df['Type_of_vehicle'] = df['Type_of_vehicle'].str.strip()
        df['Festival'] = df['Festival'].str.strip()
        df['City'] = df['City'].str.strip()

        df.drop(columns=['ID', 'Delivery_person_ID'],axis=1,inplace=True)
        
        logging.info("Values extracted sucessfully")
        return df

class UpdateDatatypesStrategy(FeatureExtractionStrategy):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Data Tyepe coversition started")

        df['Delivery_person_Age'] = df['Delivery_person_Age'].astype('int32')
        df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype('float64')
        df['multiple_deliveries'] = df['multiple_deliveries'].astype("int32")
        df['Order_Date'] = pd.to_datetime(df['Order_Date'], format="%d-%m-%Y")
        return df

class ExtractDateFeaturesStrategy(FeatureExtractionStrategy):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["day"] = df.Order_Date.dt.day
        df["month"] = df.Order_Date.dt.month
        df["quarter"] = df.Order_Date.dt.quarter
        df["year"] = df.Order_Date.dt.year
        df['day_of_week'] = df.Order_Date.dt.day_of_week.astype(int)
        df["is_month_start"] = df.Order_Date.dt.is_month_start.astype(int)
        df["is_month_end"] = df.Order_Date.dt.is_month_end.astype(int)
        df["is_quarter_start"] = df.Order_Date.dt.is_quarter_start.astype(int)
        df["is_quarter_end"] = df.Order_Date.dt.is_quarter_end.astype(int)
        df["is_year_start"] = df.Order_Date.dt.is_year_start.astype(int)
        df["is_year_end"] = df.Order_Date.dt.is_year_end.astype(int)
        df['is_weekend'] = np.where(df['day_of_week'].isin([5, 6]), 1, 0)
        return df

class CalculateTimeDiffStrategy(FeatureExtractionStrategy):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Time_Orderd'] = pd.to_timedelta(df['Time_Orderd'])
        df['Time_Order_picked'] = pd.to_timedelta(df['Time_Order_picked'])
        df['Time_Order_picked_formatted'] = df['Order_Date'] + np.where(df['Time_Order_picked'] < df['Time_Orderd'], pd.DateOffset(days=1), pd.DateOffset(days=0)) + df['Time_Order_picked']
        df['Time_Ordered_formatted'] = df['Order_Date'] + df['Time_Orderd']
        df['Time_Order_picked_formatted'] = pd.to_datetime(df['Time_Order_picked_formatted'])
        df['order_prepare_time'] = (df['Time_Order_picked_formatted'] - df['Time_Ordered_formatted']).dt.total_seconds() / 60
        df['order_prepare_time'].fillna(df['order_prepare_time'].median(), inplace=True)
        df.drop(['Time_Orderd', 'Time_Order_picked', 'Time_Ordered_formatted', 'Time_Order_picked_formatted', 'Order_Date'], axis=1, inplace=True)
        return df

class CalculateDistanceStrategy(FeatureExtractionStrategy):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df['distance'] = np.zeros(len(df))
        restaurant_coordinates = df[['Restaurant_latitude', 'Restaurant_longitude']].to_numpy()
        delivery_location_coordinates = df[['Delivery_location_latitude', 'Delivery_location_longitude']].to_numpy()
        df['distance'] = np.array([geodesic(restaurant, delivery).km for restaurant, delivery in zip(restaurant_coordinates, delivery_location_coordinates)])
        df['distance'] = df['distance'].astype("str").apply(lambda X:X.split(".")[0]).astype('int')
        return df


class FeatureExtractionHandler:
    def __init__(self, strategy: FeatureExtractionStrategy):
        """
        Initializes the FeatureExtractionPipeline with specific feature extraction strategies.

        Parameters:
        strategy (FeatureExtractionStrategy): The strategy to be added.
        """
        self._strategie = strategy

    def set_strategy(self, strategy: FeatureExtractionStrategy):
        """
        Adds a new strategy to the pipeline.

        Parameters:
        strategy (FeatureExtractionStrategy): The strategy to be added.
        """
        self._strategie.append(strategy)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes all feature extraction strategies on the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with features extracted.
        """
        #for strategy in self._strategies:
            #df = strategy.transform(df)
        return self._strategie.transform(df=df)
