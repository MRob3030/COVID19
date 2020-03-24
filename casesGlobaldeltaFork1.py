#!/usr/bin/env python
# coding: utf-8

# # Cases reported 


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cbook as cbook




date = list(pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", nrows=0))
cases = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv") 
data = np.array(cases)
dateg = date[4:]



dateend = len(date)
sizedate = len(date) -1 
sizedata = len(data) -1

datanum = data[:, 4:]

Globsum = data[1,:]
Globdelta = np.empty([sizedata,sizedate])
Globdeltasum = np.empty([5,sizedate])
Globdeltasum[0] = Globdelta[0]


# Sum the cases reports in each country k  for each date y
Globsum =   datanum.sum(axis=0)   


# Subtract the pervious day's cases in each country k  for each date y

for k in range(0, sizedata):
    for y in range (5,sizedate):
        
        Globdelta[k,y] = data[k,y] - data[k,y-1]
        Globdeltasum[0,y] = Globdeltasum[0,y] + Globdelta[k,y]
    



Globdeltas = list(Globdeltasum[0])
Globdeltasuml = Globdeltas[3:]



dateg = date[4:]

# Create dates for x axis using the first and last date in the data
firstdate = dateg[0]
lastdate = dateg[(len(dateg) -1)]
datelist = np.arange(0, len(dateg), step = 20)

seconddate = dateg[datelist[1]]
thriddate = dateg[datelist[2]]
fourthdate = dateg[datelist[3]]

plt.subplot(121)
plt.bar(dateg, Globdeltasuml)


plt.ylabel('Confirmed New Cases')
plt.xlabel('Date')
plt.title('Global New Cases Per Day')
plt.xticks(np.arange(0, len(dateg), step=20), [firstdate, seconddate, thriddate , fourthdate,  lastdate])

plt.show()


plt.subplot(122)

plt.plot(Globsum)

plt.ylabel('Confirmed New Cases')
plt.xlabel('Date')
plt.title('Global Running Total of Confirmed Cases')
plt.xticks(np.arange(0, len(dateg), step=20), [firstdate, seconddate, thriddate , fourthdate,  lastdate])

plt.show()




SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title









