import pandas as pd
import numpy as np
import time

from libs.plots import plot_array_blobs, plot_forecast
from libs.dFManipulations import parser, getAreaofDF
from libs.mathHelper import *

from sklearn.cluster import MiniBatchKMeans

def runMiniBatchKMeans(dataFrame, test_data):
    x_date = np.array(dataFrame.index.values)
    dates_of_forecast = create7dayArray(x_date[-1])
    y = np.array(dataFrame)
    n_clusters = 20

    # Timing measurement
    start_timer = time.time()

    minibatch = MiniBatchKMeans(n_clusters=n_clusters, max_iter=100, batch_size = 100, random_state=0)
    minibatch.fit(y)
    labels = minibatch.labels_

    # Timing measurement
    end_timer = time.time()
    performance = end_timer - start_timer
    print("Time for execution: " + str(performance))

    calculateAverageSilhouette(y, labels, n_clusters)

    plot_array_blobs(x_date, y, labels)
    forecasted_clusters = predictnextNDays(labels, 7)
    forecast_values = getResults(dataFrame, labels, forecasted_clusters)

    print_forecast(forecast_values, forecasted_clusters, dates_of_forecast)
    plot_forecast(forecast_values, forecasted_clusters, dates_of_forecast, test_data)


if __name__ == "__main__":
    hour_of_day = 9
    fulldataFrame = pd.read_csv(r'resources\data.csv', index_col=0, header=None, parse_dates=True, date_parser=parser)
    dataFrame = getAreaofDF(fulldataFrame, hour_of_day-1, hour_of_day+1)
    test_data = pd.read_csv(r'resources\test_data.csv')
    test_data = pd.DataFrame(test_data, columns=[str(hour_of_day)]).to_numpy()
    runMiniBatchKMeans(dataFrame, test_data)
