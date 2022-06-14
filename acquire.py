from env import host, user, password, get_db_url
import pandas as pd 
import os


def get_titanic_data(use_cache=True):
    filename = 'titanic.csv'
    if os.path.isfile(filename) and use_cache:
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('SELECT * FROM passengers;', get_db_url('titanic_db'))
        df.to_csv(filename, index=False)
        return df


def get_iris_data(use_cache=True):
    filename = 'iris.csv'

    if os.path.isfile(filename) and use_cache:
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('''
        SELECT * FROM measurements 
        JOIN species USING(species_id);''' , get_db_url('iris_db'))
        df.to_csv(filename, index=False)
        return df


def get_telco_data(use_cache=True):
    filename = 'telco.csv'
    if os.path.isfile(filename) and use_cache:
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('''
        SELECT * FROM customers
        JOIN contract_types USING (contract_type_id)
        JOIN payment_types USING (payment_type_id)
        JOIN internet_service_types USING (internet_service_type_id);''' , get_db_url('telco_churn'))
        df.to_csv(filename, index=False)
        return df
