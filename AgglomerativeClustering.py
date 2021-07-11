import pandas as pd
import numpy as np
import time

from libs.plots import plot_aglomerative_tree
from libs.dFManipulations import parser, getPartofDF
from libs.mathHelper import create7dayArray, predictnextNDays, getResults, calculateAverageSilhouette

from sklearn.cluster import AgglomerativeClustering

def runAgglomerativeClustering(dataFrame):
    x_date = np.array(dataFrame.index.values)
    x_date_predict = create7dayArray(x_date[-1])
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
    labels_of_predicted_days = predictnextNDays(labels, 7)
    getResults(dataFrame, labels, labels_of_predicted_days, x_date_predict, area_of_values_in_a_day=False)

if __name__ == "__main__":
    fulldataFrame = pd.read_csv(r'resources\data.csv', index_col=0, header=None, parse_dates=True, date_parser=parser)
    dataFrame = getPartofDF(fulldataFrame, 9)
    runAgglomerativeClustering(dataFrame)