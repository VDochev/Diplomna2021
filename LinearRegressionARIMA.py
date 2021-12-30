import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima_model import ARIMA


from libs.dFManipulations import getPartofDF, parser
from libs.mathHelper import create7dayArray
from libs.plots import plot_error_rate, plot_poly_line


def predictARIMA(dataFrame):
    model = ARIMA(dataFrame)
    results = model.fit(disp=-1, method="css", trend="c", maxiter=500, solver="powell")
    preictedFutureValues = results.predict(start = len(dataFrame), end = len(dataFrame)+6, dynamic= True)
    preictedFutureValues = preictedFutureValues + dataFrame[-1]
    return preictedFutureValues

def runPolynomialRegressionARIMA(dataFrame, test_data, hour):
    # Use data from the dateFrame for the hour_of_day
    x_date = np.array(dataFrame.index.values)
    x = np.array(range(0, len(x_date)))
    y = np.array(dataFrame)
    degreeOfPolynomialFeatures = 6

    # Timing measurement
    start_timer = time.time()

    # Transforming the data to include another axis and use polinomials
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    polynomial_features = PolynomialFeatures(degree=degreeOfPolynomialFeatures)
    x_poly = polynomial_features.fit_transform(x)

    # Make a model and draw some function to predict the expected possible results
    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    # Prints for determining the best degree for the algorithm
    #print_calculated_errors(y, y_poly_pred)
    #plot_poly_line(x_date, y, y_poly_pred)

    # Predict using ARIMA and plot
    ARIMAprediction = predictARIMA(y_poly_pred.reshape(-1))
    prediction_line_ARIMA = np.concatenate((y_poly_pred.reshape(-1), ARIMAprediction))

    # Timing measurement
    end_timer = time.time()
    performance = end_timer - start_timer
    print("Time for execution: " + str(performance))

    x_date_predict = create7dayArray(x_date[-1])
    x_date_predict = np.concatenate((x_date, x_date_predict))

    plot_poly_line(x_date_predict, y, prediction_line_ARIMA, hour)

    error_count = []
    error_rate = 0
    day = 0
    for day in range(7):
            error_rate = float(np.abs(ARIMAprediction[day] - test_dataframe[day]) / test_dataframe[day])
            if error_rate > 1: error_rate = 1
    error_count.append(error_rate)
    
    average_error = np.average(error_count)
    print("Average accuracy: {}".format(average_error))
    return average_error

if __name__ == "__main__":
    fullDataFrame = pd.read_csv(r'resources/data_2015-20.csv', index_col=0, header=None, parse_dates=True, date_parser=parser)
    test_data = pd.read_csv(r'resources/data_2015-20_test.csv')
    error_rate = []

    for hour_of_day in range(1, 24):
        dataFrame = getPartofDF(fullDataFrame, hour_of_day)
        test_dataframe = pd.DataFrame(test_data, columns=[str(hour_of_day)]).to_numpy()
        error = runPolynomialRegressionARIMA(dataFrame, test_dataframe, hour_of_day)
        error_rate.append(error)
    
    plot_error_rate(error_rate, "LinearRegressionMovingAverage")
