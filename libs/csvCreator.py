from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import csv

holidays = [
    date(2015, 1, 1), date(2015, 3, 3), date(2015, 4, 10), date(2015, 4, 11), date(2015, 4, 12), date(2015, 4, 13), date(2015, 5, 1), date(2015, 5, 6), date(2015, 5, 24), date(2015, 9, 6), date(2015, 9, 22), date(2015, 12, 24), date(2015, 12, 25), date(2015, 12, 26), date(2015, 12, 31),
    date(2016, 1, 1), date(2016, 3, 3), date(2016, 4, 29), date(2016, 4, 30), date(2016, 5, 1), date(2016, 5, 2), date(2016, 5, 6), date(2016, 5, 24), date(2016, 9, 6), date(2016, 9, 22), date(2016, 12, 24), date(2016, 12, 25), date(2016, 12, 26), date(2016, 12, 31),
    date(2017, 1, 1), date(2017, 1, 2), date(2017, 3, 3), date(2017, 4, 14), date(2017, 4, 15), date(2017, 4, 16), date(2017, 4, 17), date(2017, 5, 1), date(2017, 5, 6), date(2017, 5, 8), date(2017, 5, 24), date(2017, 9, 6), date(2017, 9, 22), date(2017, 12, 24), date(2017, 12, 25), date(2017, 12, 26), date(2017, 12, 27), date(2017, 12, 31),
    date(2018, 1, 1), date(2018, 3, 3), date(2018, 3, 5), date(2018, 4, 6), date(2018, 4, 7), date(2018, 4, 8), date(2018, 4, 9), date(2018, 5, 1), date(2018, 5, 6), date(2018, 5, 7), date(2018, 5, 24), date(2018, 9, 6), date(2018, 9, 22), date(2018, 9, 24), date(2018, 12, 24), date(2018, 12, 25), date(2018, 12, 26),
    date(2019, 1, 1), date(2019, 3, 3), date(2019, 3, 4), date(2019, 4, 26), date(2019, 4, 27), date(2019, 4, 28), date(2019, 4, 29), date(2019, 5, 1), date(2019, 5, 6), date(2019, 5, 24), date(2019, 9, 6), date(2019, 9, 22), date(2019, 9, 23), date(2019, 12, 24), date(2019, 12, 25), date(2019, 12, 26),
    date(2020, 1, 1), date(2020, 3, 3), date(2020, 4, 17), date(2020, 4, 18), date(2020, 4, 19), date(2020, 4, 20), date(2020, 5, 1), date(2020, 5, 6), date(2020, 5, 24), date(2020, 5, 25), date(2020, 9, 6), date(2020, 9, 7), date(2020, 9, 22), date(2020, 12, 24), date(2020, 12, 25), date(2020, 12, 26), date(2020, 12, 28)
]

start_dt = datetime(2015, 1, 1, 0, 0)
end_dt = datetime(2020, 12, 31, 0, 0)
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
writer.writerow(["Date", "Consumption", "Temp [C]", "Rest Day"])

for i in range(len(consumption)):
    if date_array[i].weekday() > 5:
        rest_day = "yes"
    elif date_array[i].date() in holidays:
        rest_day = "yes"
    else:
        rest_day = "no"
    writer.writerow([date_array[i], consumption[i], temp_array[i], rest_day])

f.close()
