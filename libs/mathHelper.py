from numpy.lib.arraysetops import unique
from itertools import zip_longest
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from scipy.stats import beta
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score

def get_rmse(x, x_poly_pred):
    return np.sqrt(mean_squared_error(x, x_poly_pred))

def get_r2(x, x_poly_pred):
    return r2_score(x, x_poly_pred)

def print_calculated_errors(x, x_poly_pred):
    print("RMSE of polynomial regression is: ", get_rmse(x, x_poly_pred))
    print("R2 of polynomial regression is ", get_r2(x, x_poly_pred))

def print_stationarity(dataFrame):
    # Dickey–Fuller test:
    result = adfuller(dataFrame)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

def print_forecast(forecast_values, labels_of_forecast, dates_of_forecast):
    day = 0
    for label in labels_of_forecast:
        print("Day: " + str(dates_of_forecast[day]), end=', ')
        print("Expected min: %d, Expected max: %d, Expected average: %.2f" % forecast_values[label])
        day += 1

def create7dayArray(lastDay):
    days = 7
    nextSevenDays = np.array([lastDay + np.timedelta64(i,'D') for i in range(1, days+1)])
    return nextSevenDays

def getBetaDistCoef(window_size):
    x = np.linspace(0, 1, window_size + 2)
    y = beta.pdf(x, 0.1, 0.9)
    return y[1:-1]

def getDayPrediction(window):
    beta = getBetaDistCoef(len(window))
    unique_values = np.unique(window)
    coef_calc = [0] * len(unique_values)

    for idx in range(len(window)):
        for uq_idx in range(len(unique_values)):
            if window[idx] == unique_values[uq_idx]:
                coef_calc[uq_idx] += beta[idx]

    return unique_values[coef_calc.index(max(coef_calc))]

def predictnextNDays(dataFrame, window_size):
    window = dataFrame[-window_size:]
    result = []
    for idx in range(window_size):
        temp_window = window[idx:idx+window_size]
        windowCalculation = getDayPrediction(temp_window)
        window = np.append(window, windowCalculation)
        result = np.append(result, windowCalculation)
    return result

def predictRollingWindow(dataFrame, window_size):
    window = dataFrame[:window_size]
    result = []
    for idx in range(len(dataFrame)-window_size):
        window = dataFrame[idx:idx+window_size]
        windowCalculation = getDayPrediction(window)
        result = np.append(result, windowCalculation)
    return result

def getDaysInCluster(dataFrame, labels, cluster):
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = dataFrame.index.values
    cluster_map['cluster'] = labels
    return cluster_map[cluster_map.cluster == cluster]['data_index']

def calculateMinMaxAverage(dataFrame):
    min_v = dataFrame.min()
    max_v = dataFrame.max()
    average_v = dataFrame.mean()
    return min_v, max_v, average_v

def getResults(dataFrame, labels, forecasted_clusters):
    forecast_values = {}
    for cluster in unique(forecasted_clusters):
        values_in_cluster = []
        days_in_cluster = getDaysInCluster(dataFrame, labels, cluster)
        for day in days_in_cluster:
            values_in_cluster = np.append(values_in_cluster, dataFrame.loc[day])
        forecast_values[cluster] = calculateMinMaxAverage(values_in_cluster)
    return forecast_values

def calculateAverageSilhouette(X, labels, n_clusters):
    silhouette_avg = silhouette_score(X, labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)