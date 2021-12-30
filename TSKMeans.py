import pandas as pd
import numpy as np
import time

#from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans

from libs.plots import plot_array_blobs, plot_forecast, plot_error_rate
from libs.dFManipulations import parser, getAreaofDF
from libs.mathHelper import *

def runTSKMeans(dataFrame, test_data, hour):
    x_date = np.array(dataFrame.index.values)
    dates_of_forecast = create7dayArray(x_date[-1])
    y = np.array(dataFrame)
    n_clusters = 25

    # Timing measurement
    start_timer = time.time()

    # KMeans algorithm for time series
    tskmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="softdtw", max_iter=10)
    tskmeans.fit(y)
    labels = tskmeans.labels_

    # Timing measurement
    end_timer = time.time()
    performance = end_timer - start_timer
    print("Time for execution: " + str(performance))

    calculateAverageSilhouette(y, labels, n_clusters)

    plot_array_blobs(x_date, y, labels)
    forecasted_clusters = predictnextNDays(labels, 7)
    forecast_values = getResults(dataFrame, labels, forecasted_clusters)

    print_forecast(forecast_values, forecasted_clusters, dates_of_forecast)
    plot_forecast(forecast_values, forecasted_clusters, dates_of_forecast, test_data, hour)

    error_count = []
    error_rate = 0
    day = 0
    for label in forecasted_clusters:
        if test_data[day] < forecast_values[label][1] and test_data[day] > forecast_values[label][0]:
            error_rate = 0.
        else:
            error_rate = float(np.abs(forecast_values[label][2] - test_data[day]) / test_data[day])
        day += 1
        error_count.append(error_rate)
    
    average_error = np.average(error_count)
    print("Average accuracy: {}".format(average_error))
    return average_error

if __name__ == "__main__":
    fulldataFrame = pd.read_csv(r'resources/data_2015-20.csv', index_col=0, header=None, parse_dates=True, date_parser=parser)
    test_data = pd.read_csv(r'resources/data_2015-20_test.csv')
    
    error_rate = []
    for hour_of_day in range(2, 24):
        dataFrame = getAreaofDF(fulldataFrame, hour_of_day-1, hour_of_day+1)
        test_data_2 = pd.DataFrame(test_data, columns=[str(hour_of_day)]).to_numpy()
        error = runTSKMeans(dataFrame, test_data_2, hour_of_day)
        error_rate.append(error)
    
    plot_error_rate(error_rate, "TSKMeans")