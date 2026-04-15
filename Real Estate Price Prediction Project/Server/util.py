import json
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


__locations = None
__data_columns = None
__model = None

# predict_price function to remove invalid feature names warning:

def predict_price(model, X_columns, input_values):
    input_df = pd.DataFrame([input_values], columns = X_columns)
    return model.predict(input_df[0])

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def get_location_names():
    return __locations
    
def load_saved_artifacts():
    print("loading saved artifacts........ start")
    global __data_columns
    global __locations
    global __model

    with open("./Artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open("./Artifacts/banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts..... done!")


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    
    print(get_estimated_price('hoodi', 800, 3, 3))
    print(get_estimated_price('hoodi', 800, 2, 2))
    print(get_estimated_price('frazer town', 1500, 1, 2))
    print(get_estimated_price('indira nagar', 1300, 2, 2))