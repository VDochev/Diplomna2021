import pandas as pd
import numpy as np
import time

from libs.plots import plot_array_blobs
from libs.dFManipulations import parser, getAreaofDF
from libs.mathHelper import create7dayArray, predictnextNDays, getResults, calculateAverageSilhouette

from sklearn.cluster import OPTICS

def runOPTICS(dataFrame):
    x_date = np.array(dataFrame.index.values)
    dates_to_forecast = create7dayArray(x_date[-1])
    y = np.array(dataFrame)
    min_samples = 2
    metric = "cosine"

    # Timing measurement
    start_timer = time.time()

    optics = OPTICS(min_samples=min_samples, metric=metric)
    optics.fit(y)
    labels = optics.labels_

    # Timing measurement
    end_timer = time.time()
    performance = end_timer - start_timer
    print("Time for execution: " + str(performance))

    calculateAverageSilhouette(y, labels, min_samples)

    plot_array_blobs(x_date, y, labels)
    labels_of_forecast = predictnextNDays(labels, 7)
    getResults(dataFrame, labels, labels_of_forecast, dates_to_forecast)


if __name__ == "__main__":
    fulldataFrame = pd.read_csv(r'resources\data.csv', index_col=0, header=None, parse_dates=True, date_parser=parser)
    dataFrame = getAreaofDF(fulldataFrame, 8, 10)
    runOPTICS(dataFrame)
