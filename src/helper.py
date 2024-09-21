import numpy as np
import pandas as pd
import os,pickle

raw_data = pd.read_csv('data/train.csv')
raw_data.rename(columns={'Weatherconditions': 'Weather_conditions'},inplace=True)
column_strategies = {
            'Delivery_person_Age': ('constant', [np.random.choice(raw_data['Delivery_person_Age'])]),  
            'Weather_conditions': ('constant', [np.random.choice(raw_data['Weather_conditions'])]),  
            'City': ('mode', None),
            'Festival': ('mode', None),
            'multiple_deliveries': ('mode', None),
            'Road_traffic_density': ('mode', None),
            'Delivery_person_Ratings': ('median', None)
            }
"""
model_file_path = os.path.join("artifacts","model.pkl")
    with open(model_file_path,'wb') as f:
        pickle.dump(model,f)
"""