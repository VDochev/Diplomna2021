from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import time

import libs.plots as plots
import libs.dFManipulations as dfman
import libs.mathHelper as mhelp

from sklearn.cluster import AgglomerativeClustering

def runAgglomerativeClustering(dataFrame, test_data, hour):
    x_date = np.array(dataFrame.index.values)
    dates_of_forecast = mhelp.create7dayArray(x_date[-1])
    y = np.array(dataFrame)
    n_clusters = 25

    # Timing measurement
    start_timer = time.time()

    agglomerativeClustering = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerativeClustering.fit(y.reshape(-1, 1))
    labels = agglomerativeClustering.labels_

    # Timing measurement
    end_timer = time.time()
    performance = end_timer - start_timer
    print("Time for execution: " + str(performance))

    mhelp.calculateAverageSilhouette(y.reshape(-1, 1), labels, n_clusters)

    plots.plot_aglomerative_tree(x_date, y, labels)
    forecasted_clusters = mhelp.predictnextNDays(labels, 7)
    forecast_values = mhelp.getResults(dataFrame, labels, forecasted_clusters)

    mhelp.print_forecast(forecast_values, forecasted_clusters, dates_of_forecast)
    plots.plot_forecast(forecast_values, forecasted_clusters, dates_of_forecast, test_data, hour)
    error_count = []
    error_rate = 0
    day = 0
    for label in forecasted_clusters:
        if test_data[day] < forecast_values[label][1] and test_data[day] > forecast_values[label][0]:
            error_rate = 0
        else:
            error_rate = (forecast_values[label][2] - test_data[day]) / test_data[day]
        day += 1
        error_count.extend(error_rate)
    
    average_error = np.average(error_count)
    print("Average accuracy: {}".format(average_error))
    return average_error

if __name__ == "__main__":
    hour_of_day = 9
    fulldataFrame = pd.read_csv(r'resources/data_2015-20.csv', index_col=0, header=None, parse_dates=True, date_parser=dfman.parser)
    test_data = pd.read_csv(r'resources/data_2015-20_test.csv')
    error_rate = []
    for hour in range(1, 24):
        dataFrame = dfman.getPartofDF(fulldataFrame, hour)
        test_data_2 = pd.DataFrame(test_data, columns=[str(hour)]).to_numpy()
        error = runAgglomerativeClustering(dataFrame, test_data_2, hour)
        error_rate.append(error)
    
    plots.plot_error_rate(error_rate, "AgglomerativeClustering")
