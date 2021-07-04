import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters

from libs.plots import plot_csv
from libs.dFManipulations import parser, getPartofDF
from libs.mathHelper import print_stationarity, create7dayArray

def calculate_ARIMA_result(df_log):
    model = ARIMA(df_log, order=(4,1,4))
    results = model.fit(disp=-1, method="css", trend="c", maxiter=500, solver="powell")
    return results

def predict_ARIMA(df_log, ARIMA_results):
    predictions_ARIMA_diff = pd.Series(ARIMA_results.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(df_log.iloc[0], index=df_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    return predictions_ARIMA

def runARIMA(dataFrame):
    # Get logarithm with base e of dataFrame
    df_log = np.log(dataFrame)
    print(df_log)
    x_date = np.array(dataFrame.index.values)
    x_date_predict = create7dayArray(x_date[-1])

    ARIMA_result = calculate_ARIMA_result(df_log)
    prediction = predict_ARIMA(df_log, ARIMA_result)

    plt.plot(dataFrame)
    plt.plot(prediction)
    #ARIMA_result.plot_predict(start=64, end=70, alpha=0.5)
    ARIMA_result.plot_predict(alpha=0.5)
    plt.show()

if __name__ == "__main__":
    register_matplotlib_converters()
    fullDataFrame = pd.read_csv(r'resources\data.csv', index_col=0, header=0, parse_dates=True, date_parser=parser)
    dataFrame = getPartofDF(fullDataFrame, 1)
    runARIMA(dataFrame)

