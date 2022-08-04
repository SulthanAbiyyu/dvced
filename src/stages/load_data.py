import pandas as pd

def load_data(config):
    data = pd.read_csv(config['data_load']['raw_data'], delimiter=";")
    return data

