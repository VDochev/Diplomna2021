from datetime import datetime

def getPartofDF(fulldataFrame, hour_of_the_day):
    return getAreaofDF(fulldataFrame, hour_of_the_day, hour_of_the_day)

def getAreaofDF(fulldataFrame, first_hour, last_hour):
    if first_hour == last_hour:
        result = fulldataFrame.iloc[:,first_hour]
    else:
        result = fulldataFrame.iloc[:,first_hour-1:last_hour]
    return result

def parser(x):
    return datetime.strptime(x, '%d.%m.%Y').date()

def addFeaturesToData(dataFrame, features):
    return dataFrame.join(features)

def removeFeatureFromData(dataFrame, hour_of_the_day):
    return dataFrame.iloc[:,:hour_of_the_day]