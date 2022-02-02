import urllib.request
from datetime import datetime, timedelta

import csv
import bs4 as bs
import numpy as np
from scipy import signal as sig

start_dt = datetime(2015, 1, 1, 0, 0)
end_dt = datetime(2021, 1, 1, 0, 0)
day_count = (end_dt - start_dt).days + 1
date_array = [start_dt + timedelta(hours = n) for n in range(day_count * 24)]

consumption = []
f_consump = open('resources/data_2015-20.csv', 'r', newline='')
csvreader_cons = csv.reader(f_consump)
for row in csvreader_cons:
    consumption.extend(row[1:])
f_consump.close()

f_consump = open('resources/data_2015-20_test.csv', 'r', newline='')
csvreader_cons = csv.reader(f_consump)
_ = next(csvreader_cons)
for row in csvreader_cons:
    consumption.extend(row[1:])
f_consump.close()

temp_array = []
f_temp = open('resources/temperature.csv', 'r', newline='')
csvreader_temp = csv.reader(f_temp)
_ = next(csvreader_temp)
for row in csvreader_temp:
    temp_array.append(row[1])
f_temp.close()

f = open('resources/full.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(["Date", "Consumption", "Temp [C]"])

for i in range(len(consumption)):
    writer.writerow([date_array[i], consumption[i], temp_array[i]])

f.close()
