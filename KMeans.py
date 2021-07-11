import pandas as pd
import numpy as np
import time

from sklearn.cluster import KMeans

from libs.plots import plot_array_blobs
from libs.dFManipulations import parser, getAreaofDF
from libs.mathHelper import create7dayArray, predictnextNDays, getResults, calculateAverageSilhouette

def runKMeans(dataFrame):
    x_date = np.array(dataFrame.index.values)
    dates_to_forecast = create7dayArray(x_date[-1])
    y = np.array(dataFrame)
    n_clusters = 25

    # Timing measurement
    start_timer = time.time()

    # Standart KMeans algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(y)
    labels = kmeans.labels_

    # Timing measurement
    end_timer = time.time()
    performance = end_timer - start_timer
    print("Time for execution: " + str(performance))

    calculateAverageSilhouette(y, labels, n_clusters)

    plot_array_blobs(x_date, y, labels)
    labels_of_predicted_days = predictnextNDays(labels, 7)
    getResults(dataFrame, labels, labels_of_predicted_days, dates_to_forecast)

if __name__ == "__main__":
    fulldataFrame = pd.read_csv(r'resources\data.csv', index_col=0, header=None, parse_dates=True, date_parser=parser)
    dataFrame = getAreaofDF(fulldataFrame, 8, 10)
    runKMeans(dataFrame)
