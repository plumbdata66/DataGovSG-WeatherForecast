# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 00:21:29 2017

@author: zx_pe
"""
import pickle
import requests
import pandas as pd
from scipy.spatial.distance import euclidean
import datetime
import numpy as np
    


def calldata(date_time, api_key, resource_list):
    params = {'date_time': date_time}
    headers = {
                'api-key': api_key,
                'accept' : 'application/json'
               }
    x = {url.split('/')[-1]: requests.get(url, headers = headers, params = params).json() for url in resource_list}
    return x

def toDataFrame(x):
    d, t = x['rainfall']['items'][0]['timestamp'] .split('T')
    print(d,t)
    values = {}
    values['rainfall'] =  {item['station_id']: item['value'] for item in x['rainfall']['items'][0]['readings']}    
    values['windspeed'] = {item['station_id']: item['value'] for item in x['wind-speed']['items'][0]['readings']} #station IDs  
    values['airtemp'] =  {item['station_id']: item['value'] for item in x['air-temperature']['items'][0]['readings']} #station IDs
    try:
        values['pm25'] = x['pm25']['items'][0]['readings']['pm25_one_hourly'] #central east north south west, dict
    except:
        values['pm25'] = {'NoStation': np.nan}
#    values['psi'] = pd.DataFrame(x['psi']['items'][0]['readings']) #Many types of readings, loop into key/value pairs, central east north south west national
    
    values['humidity'] = {item['station_id']: item['value'] for item in x['relative-humidity']['items'][0]['readings']}
    values['winddir'] =  {item['station_id']: item['value'] for item in x['wind-direction']['items'][0]['readings']} #station IDs
    
    
    try:
        psi = x['psi']['items'][0]['readings']
        psi_cols = list(psi.keys())
        values = {**values, **psi} #Many types of readings, loop into key/value pairs, central east north south west national
        nearest = findNearestStation(x)
        attr = pd.DataFrame.from_dict(nearest, orient='index')
        attr.columns = ['windspeed', 'airtemp', 'rainfall', 'winddir', 'humidity', 'pm25']
    
        for col in psi_cols:
            attr[col] = attr['pm25']
    except:
        print('No psi readings')
        nearest = findNearestStation(x)
        attr = pd.DataFrame.from_dict(nearest, orient='index')
        attr.columns = ['windspeed', 'airtemp', 'rainfall', 'winddir', 'humidity', 'pm25']

        
    for v in values:
        attr[v] = [values[v][i] for i in attr[v]]
    attr['datestamp'] = d + '/' + str(np.floor(int(t.split(':')[0])/6))     
    return attr
#%% Reorg dictionaries
def nearestStation(loc, locList):
    if len(locList) == 0:
        return 'NoStation'
    l = sorted([(entry[0], euclidean(loc,(entry[1],entry[2]))) for entry in locList], key = lambda d: d[1])
    return l[0][0]

def findNearestStation(x):
    windspeed_locs = [(entry['id'],
                       entry['location']['latitude'],
                      entry['location']['longitude']) for entry in x['wind-speed']['metadata']['stations']] 
        
    airtemp_locs = [(entry['id'],
                       entry['location']['latitude'],
                      entry['location']['longitude']) for entry in x['air-temperature']['metadata']['stations']]
        
    rainfall_locs = [(entry['id'],
                       entry['location']['latitude'],
                      entry['location']['longitude']) for entry in x['rainfall']['metadata']['stations']]    
        
    winddir_locs = [(entry['id'],
                       entry['location']['latitude'],
                      entry['location']['longitude']) for entry in x['wind-direction']['metadata']['stations']]     
        
    humidity_locs = [(entry['id'],
                       entry['location']['latitude'],
                      entry['location']['longitude']) for entry in x['relative-humidity']['metadata']['stations']]     
        
    pm25_locs = [(entry['name'], entry['label_location']['latitude'], entry['label_location']['longitude'])for entry in x['pm25']['region_metadata']]
    
    locations = [windspeed_locs, airtemp_locs, rainfall_locs, winddir_locs, humidity_locs, pm25_locs]
    nearest = {station[0]: [nearestStation(station[1:3], locList) for locList in locations] for station in rainfall_locs}
    
    return nearest
    
#%%

def scraper(datestart, dateend, api_key, resourcelist, by = {'day': 1}):
    datelabels = []
    response = []
    Datestart = datetime.datetime.strptime(datestart, '%Y-%m-%d')
    Dateend = datetime.datetime.strptime(dateend, '%Y-%m-%d')
    while Datestart < Dateend:
        print(Datestart)
        datelabels.append(Datestart)
        r = calldata(Datestart.isoformat(), api_key, resourcelist)
        response.append(r)
        Datestart += datetime.timedelta(**by)
    
    return response

#%%
    
resourcelist = ['https://api.data.gov.sg/v1/environment/wind-speed',
               'https://api.data.gov.sg/v1/environment/air-temperature',
               'https://api.data.gov.sg/v1/environment/pm25',
               'https://api.data.gov.sg/v1/environment/psi',
               'https://api.data.gov.sg/v1/environment/rainfall',
               'https://api.data.gov.sg/v1/environment/relative-humidity',
               'https://api.data.gov.sg/v1/environment/wind-direction'
               ]
api_key = 'FebEWY2sA3BOAJIQzZsXMqrCMjVvAAuc'

datestart = '2017-07-15'
dateend = '2017-09-15'
response = scraper(datestart, dateend, api_key, resourcelist)

#pickle.dump(response, open('response.p', 'wb'))
#%% 
def rainDataFrame(x):
    d, t = x['rainfall']['items'][0]['timestamp'] .split('T')
    print(d)
    values = {}
    values['rainfall'] =  {item['station_id']: item['value'] for item in x['rainfall']['items'][0]['readings']}    

    attr = pd.DataFrame(list(values['rainfall'].items()), columns = ['SID','rainfall'])
    
    attr['datestamp'] = d + '/' + str(np.floor(int(t.split(':')[0])/6))
    return attr
#%% Scrape hourly rainfall data
datestart = '2017-01-01'
dateend = '2017-01-15'
rainfall =  scraper(datestart, dateend, api_key, resourcelist =['https://api.data.gov.sg/v1/environment/rainfall'], by = {'hours': 1})
rain = pd.concat(rainfall)
rain = rain.groupby(['SID','datestamp']).sum().groupby(['SID','datestamp']).sum()
raining = [rain]

for i in range(2,9):
    datestart = '2017-0{}-15'.format(i)
    dateend = '2017-0{}-15'.format(i+1)
    rainfall = scraper(datestart, dateend, api_key, resourcelist =['https://api.data.gov.sg/v1/environment/rainfall'], by = {'hours': 1})
    rain = pd.concat([rainDataFrame(x) for x in rainfall])
    rain = rain.groupby(['SID','datestamp']).sum().groupby(['SID','datestamp']).sum()
    raining.append(rainfall)

#%% Combine datasets
Rain = pd.concat(raining)
pickle.dump(Rain, open('rain.p', 'wb'))

d = [toDataFrame(x) for x in response]
df = pd.concat(d)
df.drop('rainfall', axis = 1, inplace=True)
df['SID'] = df.index
Rain['SID'] = Rain.index.droplevel(1)
Rain['datestamp'] = Rain.index.droplevel(0)
df = df.merge(Rain, on = ['SID','datestamp'])

#pickle.dump(df, open('dataarray.p','wb'))
