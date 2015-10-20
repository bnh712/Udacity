# Udacity
import numpy as np
import pandas
import statsmodels.api as sm

dataframe = pandas.read_csv('turnstile_data_master_with_weather.csv')

def linear_regression(features, values):
    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]

    print intercept, params

def predictions(dataframe):
    features = dataframe[['rain', 'precipi', 'Hour', 'fog']]
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix = 'unit')
    features = features.joic(dummy_units)

    values = dataframe['ENTRIESn_hourly']

    intercept, params = linear_regression(features, values)

    predictions = intercept + np.dot(features, params)

    return predictions

