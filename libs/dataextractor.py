import urllib.request
from datetime import datetime, timedelta, date

import csv
import bs4 as bs
import numpy as np
from scipy import signal as sig

f = open('resources/temperature.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(["Date", "Temp [C]"])

start_dt = datetime(2015, 1, 1, 0, 0)
end_dt = datetime(2021, 1, 1, 0, 0)

day_count = (end_dt - start_dt).days + 1
for single_date in (start_dt + timedelta(n) for n in range(day_count)):
    url = "https://www.stringmeteo.com/synop/bg_stday.php?year={}&month={}&day={}&city=15614&int=1&submit=%D0%9F%D0%9E%D0%9A%D0%90%D0%96%D0%98#sel".format(single_date.year, single_date.month, single_date.day)
    soup = bs.BeautifulSoup(urllib.request.urlopen(url).read())

    min_values=[]
    max_values=[]
    avg_values=[]

    for sup in soup.findAll('table')[0].findAll(attrs={'class' : 'red'}):
            if "C" in sup.text:
                try:
                    min_values.append(float(sup.text[:-2]))
                except ValueError:
                    print("Error: No valid string(red): {} {}".format(single_date, sup.text))
                    continue

    for sup in soup.findAll('table')[0].findAll(attrs={'class' : 'blue'}):
            if "C" in sup.text:
                try:
                    max_values.append(float(sup.text[:-2]))
                except ValueError:
                    print("Error: No valid string(blue): {} {}".format(single_date, sup.text))
                    continue
    
    min_values = min_values[:-2]
    max_values = max_values[:-2]

    for minv, maxv in zip(min_values, max_values):
        avg_values.append(float((minv + maxv) / 2))

    if len(avg_values) < 4:
        print("Error: No values: {} {}".format(single_date, len(avg_values)))
        avg_values = np.empty((24))
        avg_values[:] = np.NaN
    else:
        avg_values = sig.resample(avg_values, 24)

    delta2 = timedelta(hours = 1)
    dates = [single_date + delta2 * x for x in range(24)]
    for i in range(24):
        writer.writerow([dates[i], avg_values[i]])
f.close()