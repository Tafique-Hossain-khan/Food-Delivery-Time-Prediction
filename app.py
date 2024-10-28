
import pickle
import pandas as pd
import streamlit as st
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder
import requests,os
from src.logging.logger import logging
from dotenv import load_dotenv
import mlflow



def extract_date_features(data):
    data["day"] = data.Order_Date.dt.day
    data["month"] = data.Order_Date.dt.month
    data["quarter"] = data.Order_Date.dt.quarter
    data["year"] = data.Order_Date.dt.year
    data['day_of_week'] = data.Order_Date.dt.day_of_week.astype(int)
    data["is_month_start"] = data.Order_Date.dt.is_month_start.astype(int)
    data["is_month_end"] = data.Order_Date.dt.is_month_end.astype(int)
    data["is_quarter_start"] = data.Order_Date.dt.is_quarter_start.astype(int)
    data["is_quarter_end"] = data.Order_Date.dt.is_quarter_end.astype(int)
    data["is_year_start"] = data.Order_Date.dt.is_year_start.astype(int)
    data["is_year_end"] = data.Order_Date.dt.is_year_end.astype(int)
    data['is_weekend'] = np.where(data['day_of_week'].isin([5,6]),1,0)


def calculate_time_diff(df):
    # Find the difference between ordered time & picked time
    df['Time_Orderd'] = pd.to_timedelta(df['Time_Orderd'])
    
    df['Time_Order_picked'] = pd.to_timedelta(df['Time_Order_picked'])
    
    df['Time_Order_picked_formatted'] = df['Order_Date'] + np.where(df['Time_Order_picked'] < df['Time_Orderd'], pd.DateOffset(days=1), pd.DateOffset(days=0)) +     df['Time_Order_picked']
    df['Time_Ordered_formatted'] = df['Order_Date'] + df['Time_Orderd']

    df['Time_Order_picked_formatted']=pd.to_datetime(df['Time_Order_picked_formatted'])
    
    df['order_prepare_time'] = (df['Time_Order_picked_formatted'] - df['Time_Ordered_formatted']).dt.total_seconds() / 60
    
    # Handle null values by filling with the median
    #df['order_prepare_time'].fillna(df['order_prepare_time'].median(), inplace=True)
    
    # Drop all the time & date related columns
    df.drop(['Time_Orderd', 'Time_Order_picked', 'Time_Ordered_formatted', 'Time_Order_picked_formatted', 'Order_Date'], axis=1, inplace=True)



#Calculate distance between restaurant location & delivery location
def calculate_distance(df):
    df['distance']=np.zeros(len(df))
    restaurant_coordinates=df[['Restaurant_latitude','Restaurant_longitude']].to_numpy()
    delivery_location_coordinates=df[['Delivery_location_latitude','Delivery_location_longitude']].to_numpy()
    df['distance'] = np.array([geodesic(restaurant, delivery) for restaurant, delivery in zip(restaurant_coordinates, delivery_location_coordinates)])
    df['distance'] = df['distance'].astype("str").apply(lambda X:X.split(".")[0]).astype('int')
    



# Define Streamlit app
st.title('Delivery Time Prediction')

# User input for each feature
#delivery_person_id = st.text_input('Delivery Person ID', 'BANGRES19DEL01')
age = st.number_input('Delivery Person Age', min_value=18, max_value=65, value=30)
ratings = st.number_input('Delivery Person Ratings', min_value=1.0, max_value=5.0, value=4.5)
order_date = st.date_input('Order Date')
time_ordered = str(st.time_input('Time Ordered'))
time_order_picked = str(st.time_input('Time Order Picked'))
weather = st.selectbox('Weather Conditions', ['Sunny', 'Cloudy', 'Rainy', 'Foggy'])
traffic = st.selectbox('Road Traffic Density', ['Low', 'Medium', 'High', 'Jam'])
vehicle_condition = st.number_input('Vehicle Condition', min_value=0, max_value=10, value=7)
order_type = st.selectbox('Type of Order', ['Snack', 'Meal', 'Drinks', 'Buffet'])
vehicle_mapping = {
    'Bike': 'motorcycle',
    'Scooter': 'scooter',
    'Electric Scooter': 'electric_scooter',
    'Cycle': 'bicycle'
}
vehicle_type = st.selectbox('Type of Vehicle', ['Bike', 'Scooter', 'Electric Scooter', 'Cycle'])
multiple_deliveries = st.number_input('Multiple Deliveries', min_value=0, max_value=5, value=0)
festival = st.selectbox('Festival', ['No', 'Yes'])
city = st.selectbox('City', ['Urban', 'Semi-Urban', 'Metropolitian'])    

restaurant_address = st.text_input('Restaurant Address')
delivery_address = st.text_input('Delivery Address')


def get_lat_long_opencage(address, api_key):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={address}&key={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            latitude = data["results"][0]["geometry"]["lat"]
            longitude = data["results"][0]["geometry"]["lng"]
            return latitude, longitude
        else:
            return None, None
    else:
        return None, None

# Example usag




load_dotenv()
api_key = os.getenv('MY_API_KEY')


restaurant_lat,restaurant_long = get_lat_long_opencage(restaurant_address, api_key)
delivery_lat,delivery_long = get_lat_long_opencage(delivery_address, api_key)

def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_url});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function to set the background image
# Replace 'your_image_url' with the URL of your image or path to your local image
#set_background_image('https://zeew.eu/wp-content/uploads/2024/06/Restaurant-Online-Food-Delivery-1.webp')

if st.button("Get ETA for Delivery!"):
    # Prepare input data for the model
    input_data = pd.DataFrame({
        #'Delivery_person_ID': [delivery_person_id],
        'Delivery_person_Age': [age],
        'Delivery_person_Ratings': [ratings],
        'Restaurant_latitude': [restaurant_lat],
        'Restaurant_longitude': [restaurant_long],
        'Delivery_location_latitude': [delivery_lat],
        'Delivery_location_longitude': [delivery_long],
        'Order_Date': [order_date],
        'Time_Orderd': [time_ordered],
        'Time_Order_picked': [time_order_picked],
        'Weather_conditions': [weather],
        'Road_traffic_density': [traffic],
        'Vehicle_condition': [vehicle_condition],
        'Type_of_order': [order_type],
        'Type_of_vehicle': [vehicle_mapping[vehicle_type]],
        'multiple_deliveries': [multiple_deliveries],
        'Festival': [festival],
        'City': [city]
    })


    input_data['Order_Date'] = pd.to_datetime(input_data['Order_Date'])
    input_data['Order_Year'] = input_data['Order_Date'].dt.year
    input_data['Order_Month'] = input_data['Order_Date'].dt.month
    input_data['Order_Day'] = input_data['Order_Date'].dt.day
    #input_data['Time_Orderd'] = pd.to_datetime(input_data['Time_Orderd'], format='%H:%M:%S').dt.hour
    #input_data['Time_Order_picked'] = pd.to_datetime(input_data['Time_Order_picked'], format='%H:%M:%S').dt.hour

    logging.info(input_data.columns)
    logging.info(input_data)

    extract_date_features(input_data)
    calculate_time_diff(input_data)
    calculate_distance(input_data)
    

    input_data=input_data.drop(['Order_Year', 'Order_Month', 'Order_Day'],axis=1)

    


    encoder_file_path = os.path.join("final_model","preprocessor.pkl")
    with open(encoder_file_path,'rb') as f:
        encoder = pickle.load(f)

    model_uri = "models:/RandomForest_model/Production"  # or specify a version number
    loaded_model = mlflow.sklearn.load_model(model_uri)
    #predictions = loaded_model.predict(X_test)

    input_data.to_csv("input_data.csv",index=False)

    # Creating the DataFrame for training
    cat_col = input_data.select_dtypes(include=['object']).columns
    encoded_df = encoder.transform(input_data[cat_col])
    encoded_df_final = pd.DataFrame(encoded_df.toarray(),columns=encoder.get_feature_names_out())
    input_data.drop(columns=cat_col,axis=1,inplace=True)
    final_df = pd.concat([input_data,encoded_df_final],axis=1)
    prediction = loaded_model.predict(final_df)

    # Display the prediction
    st.write(f'Your Food will arrive in minutes): {prediction[0]}')




