from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_csv(dataFrame):
    _, ax = plt.subplots()
    plt.title('Plot full Dataframe')

    dataFrame.plot(use_index=True,
            xlabel='Day', ylabel='mW', ax=ax,
            grid=True, legend=False, marker="d")

    # Open in maximized window
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.show()

def plot_csv_scater(dataFrame):
    _, ax = plt.subplots()
    plt.title('Plot full Dataframe')
    x = dataFrame.index
    y = dataFrame.values
    ax.plot_date(x, y, xdate=True, ydate=False)
    # Open in maximized window
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.show()

def plot_poly_line(x, y, prediction_line):
    x_for_y = x[:len(y)]
    plt.scatter(x_for_y, y, s=5)
    plt.plot(x, prediction_line, color='m')
    plt.show()

def plot_array_blobs(x, y, labels):
    # Get length of y for the day and x
    x_len = len(x)
    y_per_day_len = y.shape[1]

    plt.figure()
    # Create array of random colors
    colors = cm.rainbow(np.linspace(0, 1, max(labels)+1))
    # Plot with different colors based on the lable
    for i in range(x_len):
        for j in range(y_per_day_len):
            plt.plot_date(x[i], y[i][j], xdate=True, ydate=False, color=colors[labels[i]])
    plt.show()

def plot_aglomerative_tree(x, y, labels):
    # Get length of y for the day and x
    x_len = len(x)

    plt.figure()
    # Create array of random colors
    colors = cm.rainbow(np.linspace(0, 1, max(labels)+1))
    # Plot with different colors based on the lable
    for i in range(x_len):
        plt.plot_date(x[i], y[i], xdate=True, ydate=False, color=colors[labels[i]])
    plt.show()

def plot_forecast(forecast_values, labels_of_forecast, dates_of_forecast):
    day = 0
    _, ax = plt.subplots()
    for label in labels_of_forecast:
        ax.plot_date([dates_of_forecast[day]] * 3, forecast_values[label])
        day += 1
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
