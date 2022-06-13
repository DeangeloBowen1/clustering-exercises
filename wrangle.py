#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#imports for user defined functions
from env import host, user, password, get_db_url

# Imports for calculations and data frame manipulation
import math
import numpy as np
import pandas as pd

#imports for splitting data and imputing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

#Imports for creating data visualizations
import matplotlib.pyplot as plt 
import seaborn as sns

def get_zillow_data(use_cache=True):
    '''
    This function takes in no arguments, uses the imported get_db_url function to establish a connection 
    with the mysql database, and uses a SQL query to retrieve telco data creating a dataframe,
    The function caches that dataframe locally as a csv file called zillow.csv, it uses an if statement to use the cached csv
    instead of a fresh SQL query on future function calls. The function returns a dataframe with the telco data.
    '''
    filename = 'zillow.csv'

    if os.path.isfile(filename) and use_cache:
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('''
    SELECT
        prop.*,
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        landuse.propertylandusedesc,
        story.storydesc,
        construct.typeconstructiondesc
    FROM properties_2017 prop
    JOIN (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid) pred USING(parcelid)
    JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid AND pred.max_transactiondate = predictions_2017.transactiondate
    LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
    LEFT JOIN storytype story USING (storytypeid)
    LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
    WHERE prop.latitude IS NOT NULL AND prop.longitude IS NOT NULL AND transactiondate <= '2017-12-31';''' , get_db_url('zillow'))
        df.to_csv(filename, index=False)
    return df


def null_data(df):
    # displays null rows and displays them for exploration
    nulls = df.isnull().sum()
    rows = len(df)
    percent = nulls / rows 
    dataframe = pd.DataFrame({'rows_missing': nulls, 'percent_missing': percent})
    return dataframe


def null_cols(df):
    # observe null columns and displays them for exploration
    data = pd.DataFrame(df.isnull().sum(axis=1),
                      columns = ['cols_missing']).reset_index().groupby('cols_missing').count().reset_index().rename(columns = {'index':'rows'})
    data['percent_missing'] = data.cols_missing/df.shape[1]
    return data


def get_single_unit_homes(df):
    # define and obtain single unit properties
    single_unit_homes = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_unit_homes)]
    return

def handle_missing_values(df, prop_required_column = .67, prop_required_row = .75):
    # drops values that are within a certain proportion of missing values
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


# impute strategy and columns to be imputed
impute_strategy = {
    'mean' : [
    'calculatedfinishedsquarefeet',
    'finishedsquarefeet12',
    'structuretaxvaluedollarcnt',
    'taxvaluedollarcnt',
    'landtaxvaluedollarcnt',
    'taxamount',
    'lotsizesquarefeet'
],
    
    'most_frequent' : [
    'calculatedbathnbr',
    'fullbathcnt',
    'regionidcity',
    'regionidzip',
    'yearbuilt'
     ],
    
     'median' : [
     'censustractandblock'
     ]
 }


def impute_missing_data(df, impute_strategy):
    """Impute values after train validate test for dataframe integrity"""
    train, validate, test = split_data(df)
    
    for strategy, cols in impute_stragety.items():
        imputer = SimpleImputer(strategy = strategy)
        imputer.fit(train[cols])
        
        train[cols] = imputer.transform(train[cols])
        validate[cols] = imputer.transform(validate[cols])
        test[cols] = imputer.transform(test[cols])
    return train, validate, test


def prepare_zillow(df):
    '''Prepare zillow for data exploration
       split into train, test, validate'''
    df = get_single_unit_homes(df)
    df = handle_missing_values(df)
    train, validate, test = impute_missing_values(df, columns_strategy)
    return train, validate, test


def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for exploration'''
    train, validate, test = prepare_zillow(get_zillow_data())
    
    return train, validate, test


