import pandas as pd
import numpy as np
import time

from libs.plots import plot_aglomerative_tree, plot_forecast
from libs.dFManipulations import parser, getPartofDF
from libs.mathHelper import *

from sklearn.cluster import AgglomerativeClustering

def runAgglomerativeClustering(dataFrame, test_data=None):
    x_date = np.array(dataFrame.index.values)
    dates_of_forecast = create7dayArray(x_date[-1])
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

    calculateAverageSilhouette(y.reshape(-1, 1), labels, n_clusters)

    plot_aglomerative_tree(x_date, y, labels)
    forecasted_clusters = predictnextNDays(labels, 7)
    forecast_values = getResults(dataFrame, labels, forecasted_clusters)

    print_forecast(forecast_values, forecasted_clusters, dates_of_forecast)
    plot_forecast(forecast_values, forecasted_clusters, dates_of_forecast, test_data)

if __name__ == "__main__":
    hour_of_day = 9
    fulldataFrame = pd.read_csv(r'resources\data_2021.csv', index_col=0, header=None, parse_dates=True, date_parser=parser)
    dataFrame = getPartofDF(fulldataFrame, hour_of_day)
    test_data = pd.read_csv(r'resources\data_2021_test.csv')
    test_data = pd.DataFrame(test_data, columns=[str(hour_of_day)]).to_numpy()
    runAgglomerativeClustering(dataFrame, test_data)