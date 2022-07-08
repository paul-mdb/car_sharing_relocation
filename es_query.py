#!/usr/bin/env python
# coding: utf-8

# # Downloading and processing the data for one district
# 
# https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-update-by-query.html to update and clean the data ?

for zipcode in range(75012, 75021):
  print(zipcode)

  ## Imports & connection to Elasticsearch


  from elasticsearch import Elasticsearch
  from ssl import create_default_context
  import pandas as pd
  import numpy as np
  import sys
  import datetime

  with open('connect.txt') as f:
      str = f.readlines()

  es = Elasticsearch(str)


  # ### Change maximum size setting
  # By default the query returns the first 10,000 hits



  es.indices.put_settings(
      index="carshare_car_history",
      body={
          "index.max_result_window": 500000
      }

  )


  # ## Query
  # Fetch all data in given district. This should be repeated for each zipcode.


  result = es.search(
    index="carshare_car_history",
    body = {
    "size": 200000,
    "query": {

        "bool": {
          "must": {
            "match_all": {}
          },
          "filter": [
      {
      "geo_shape": {
        "ignore_unmapped": "true",
        "location": {
          "relation": "INTERSECTS",
          "shape": {
            "coordinates": [
              [
                [
                  1.87411, 49.08001
                ],
                [
                  1.87411,
                  48.64617
                ],
                [
                  2.72473,
                  48.64617
                ],
                [
                  2.72473,
                  49.08001
                ],
                [
                  1.87411,
                  49.08001
                ]
              ]
            ],
            "type": "Polygon"
          }
        }
      }
    },
    
    {
        "match_phrase": {
          "brand": "Zity"
        }    
    },
    
  {
        "match_phrase": {
          "zipcode": zipcode
        }    
    },

    {
        "range": { 
          "last_update": {
            "gte": "2020-01-01T00:00:00",
            "lte": "2022-05-09T00:00:00" # fetch data until May 10th, 2022
            }
          }}
          ]
    
      }

    }
  },
  request_timeout=30 # default timeout is 10sec
  
  )




  ## Convert data to dataframe


  df = pd.json_normalize(result['hits']['hits'])
  df = df[['_source.end__date', '_source.car_plate_number', '_source.status', '_source.group_id', '_source.duration', '_source.distance', '_source.location', '_source.zipcode', '_source.battery', '_source.end_battery', '_source.start_date']]

  ## Utils


  month_duration_dict = {"Jan": 31, "Feb": 28, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30, "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31}

  durations = np.array(list(month_duration_dict.values()))
  cum_durations = np.cumsum(durations)
  cum_dict = {i+1: cum_durations[i] for i in range(len(cum_durations))}
  cum_dict[0]=0

  week_dict = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

  monthdict = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5, "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10,  "Dec": 11}

  inv_month = {v: k for k, v in monthdict.items()}


  ## Map values


  df['_source.group_id'].replace("Zity", "Client", inplace=True) # Map Zity to Client
  df['_source.group_id'].replace("Zity Corporate", "Defleeted", inplace=True) # Map Zity Corporate to Defleeted
  df.rename(columns = {'_source.end__date':'end_date', '_source.car_plate_number':'car_plate_number', '_source.status':'status', '_source.group_id':'group_id', '_source.duration':'kibana_duration', '_source.distance':'distance', '_source.location':'location', '_source.zipcode':'zipcode', '_source.start_date':'start_date'}, inplace = True)


  # ### Splitting and creating columns


  df['delta_battery']=df['_source.end_battery']-df['_source.battery']
  df.drop('_source.end_battery', axis=1, inplace=True)
  df.drop('_source.battery', axis=1, inplace=True)
  df[['latitude', 'longitude']] = df['location'].str.split(',', expand=True)
  df.drop('location', axis=1, inplace=True)
  df['kibana_duration'] = pd.to_numeric(df['kibana_duration'], errors='coerce')
  df['kibana_duration'] = df['kibana_duration'].replace(np.nan, 0)
  df["kibana_duration"] = df["kibana_duration"].astype(int)
  #df['kibana_duration'] = df['kibana_duration'].astype(str)
  #df[['kibana_duration', 'trash']] = df['kibana_duration'].str.split('.', expand=True)
  #df.drop('trash', axis=1, inplace=True)
  df[['end_date', 'end_time']] = df['end_date'].str.split('T', expand=True)
  df[['end_time', 'trash']] = df['end_time'].str.split('.', expand=True)
  df.drop('trash', axis=1, inplace=True)
  df[['end_year', 'end_month', "end_day_number"]] = df['end_date'].str.split('-', expand=True)
  df.drop('end_date', axis=1, inplace=True)
  df[['end_hour', 'end_minutes', "end_seconds"]] = df['end_time'].str.split(':', expand=True)
  df.drop('end_time', axis=1, inplace=True)
  df.drop('end_seconds', axis=1, inplace=True)
  df[['start_date', 'start_time']] = df['start_date'].str.split('T', expand=True)
  df[['start_time', 'trash']] = df['start_time'].str.split('.', expand=True)
  df.drop('trash', axis=1, inplace=True)
  df[['start_year', 'start_month', "start_day_number"]] = df['start_date'].str.split('-', expand=True)
  df.drop('start_date', axis=1, inplace=True)
  df[['start_hour', 'start_minutes', "start_seconds"]] = df['start_time'].str.split(':', expand=True)
  df.drop('start_time', axis=1, inplace=True)
  df.drop('start_seconds', axis=1, inplace=True)


  ### Change types


  df['distance']= pd.to_numeric(df['distance'], errors='coerce')
  df['distance'] = df['distance'].fillna(0)
  df['delta_battery']= pd.to_numeric(df['delta_battery'], errors='coerce')
  df['delta_battery'] = df['delta_battery'].fillna(0)
  df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
  df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
  df['zipcode'] = pd.to_numeric(df['zipcode'], errors='coerce')
  df['kibana_duration'] = pd.to_numeric(df['kibana_duration'], errors='coerce')
  df['end_year'] = pd.to_numeric(df['end_year'], errors='coerce')
  df['end_month'] = pd.to_numeric(df['end_month'], errors='coerce')
  df['end_day_number'] = pd.to_numeric(df['end_day_number'], errors='coerce')
  df['end_hour'] = pd.to_numeric(df['end_hour'], errors='coerce')
  df['end_minutes'] = pd.to_numeric(df['end_minutes'], errors='coerce')
  df['start_year'] = pd.to_numeric(df['start_year'], errors='coerce')
  df['start_month'] = pd.to_numeric(df['start_month'], errors='coerce')
  df['start_day_number'] = pd.to_numeric(df['start_day_number'], errors='coerce')
  df['start_hour'] = pd.to_numeric(df['start_hour'], errors='coerce')
  df['start_minutes'] = pd.to_numeric(df['start_minutes'], errors='coerce')


  ## Calculate end time since 2020

  df['time_since_2020'] = (df['end_year']-2020)*365*24*60 + ((df['end_month']-1).map(cum_dict)+df['end_day_number'])*24*60 + df['end_hour']*60 + df['end_minutes']

  ## Sort by end date


  df = df.sort_values(by='time_since_2020', ascending = False)
  df.reset_index(drop=True, inplace = True)


  ## Calculation of the durations

  # Only the end dates are reliable. The start date for a given segment corresponds to the last end date for the same plate.


  def latest_segment(index):
      # returns index of the latest segment for the same plate
      
      plate_segments = list(df.loc[df['car_plate_number']==df.loc[index, 'car_plate_number']].index)
      previous_segments = plate_segments[plate_segments.index(index)+1:]
      if not len(previous_segments):
          return -1
      return previous_segments[0]


  def next_segment(index):
      plate_segments = list(df.loc[df['car_plate_number']==df.loc[index, 'car_plate_number']].index)
      if plate_segments.index(index)==0:
          return -1
      return plate_segments[plate_segments.index(index)-1]


  def duration(index):
      # returns the duration for a given index, only takes into account days and time
      
      previous_index = latest_segment(index)
      if previous_index==-1:
          return 0
      return df.loc[index, 'time_since_2020'] - df.loc[previous_index, 'time_since_2020']


  # Durations should be calculated on data not restricted to a specific district otherwise it makes no sense to look for the last occurence of the same plate as the car have travelled tto another district !

  #df['duration']=df.index.map(duration)
  df['duration']=df['kibana_duration'] 


  ## Next segment group-id column


  def next_group_id(index):
      next_index = next_segment(index)
      if next_index == -1:
          return "no next plate"
      else :
          return df.loc[next_index, 'group_id']
  df['next_group_id']=df.index.map(next_group_id)


  ## Keep only "CLIENT" Group-ids

  df = df[df['group_id']=="Client"]

  def new_status(index):
      status = df.loc[index, 'status']
      distance = df.loc[index, 'distance']
      if (status=="BOOKED" or status=="BOOKED_PARK") and distance==0:
          return "FREE"
      return status

  df['status']=df.index.map(new_status)


  # ### Missing locations BUG
  # 
  # In April, 2022, the locations are missins. The corresponding rows should be deleted


  df = df[df['latitude'].notna()]
  df = df[df['longitude'].notna()]

  df = df[df['end_year']>2000]


  def has_changed_status(index):
      last_index = latest_segment(index)
      if last_index==-1:
          return True
      return df.loc[index, 'status']!=df.loc[last_index, 'status']


  df['Status_has_changed']=df.index.map(has_changed_status)


  def new_feature(index, feature_name):
      # returns the feature value for the last segment which status didn't change
      
      if df.loc[index, 'Status_has_changed']:
          return df.loc[index, feature_name]
      else :
          index_list = list(df.loc[df['car_plate_number']==df.loc[index, 'car_plate_number']].index)
          bool_list = list(df.loc[df['car_plate_number']==df.loc[index, 'car_plate_number']]['Status_has_changed'])
          ind = index_list.index(index)
          while not bool_list[ind] and ind < len(index_list):
              ind+=1 # Looking for the last segment for which the status has changed
          return df.loc[index_list[ind], feature_name]
      
  def new_feature_cum(index, feature_name):
      # returns the sum of all values of the feature on the segments to interpolate
      if df.loc[index, 'Status_has_changed']:
          return df.loc[index, feature_name]
      else :
          value = int(df.loc[index, feature_name])
          index_list = list(df.loc[df['car_plate_number']==df.loc[index, 'car_plate_number']].index)
          bool_list = list(df.loc[df['car_plate_number']==df.loc[index, 'car_plate_number']]['Status_has_changed'])
          ind = index_list.index(index)
          while not bool_list[ind] and ind < len(index_list):
              ind+=1
              value += int(df.loc[index_list[ind], feature_name])
          return value


  features_to_change = ['latitude', 'longitude', "start_year", "start_month", "start_day_number", "start_hour", "start_minutes"]
  features_to_cumulate = ['distance', 'delta_battery']

  for name in features_to_change :
      df['new_'+name] = df.index.map(lambda x: new_feature(x, name))
      

  for name in features_to_cumulate :
      df['new_'+name] = df.index.map(lambda x: new_feature_cum(x, name))


  for name in features_to_change :
      df[name] = df['new_'+name]
      df.drop('new_'+name, axis=1, inplace=True)

  for name in features_to_cumulate :
      df[name] = df['new_'+name]
      df.drop('new_'+name, axis=1, inplace=True)

  # Delete lines
  def to_keep(index):
      next_index = next_segment(index)
      if next_index == -1 or df.loc[next_index, 'Status_has_changed']:
          return True
      return False


  df["to_keep"]=df.index.map(to_keep)
  df = df[df["to_keep"]]
  df.drop("to_keep", axis=1, inplace=True)
  df.drop("Status_has_changed", axis=1, inplace=True)


  ## Start date column

  def start_date(index):
      time = df.loc[index, 'time_since_2020'] - df.loc[index, 'kibana_duration'] # Start time since 2020
      year = 2020 + time // (365*24*60)
      time = time % (365*24*60)
      i = 0
      while cum_durations[i]< (time // (24*60)) and i < 12:
          i+=1
      month = inv_month[i]
      day = time // (24*60) - cum_durations[i-1] if i else time // (24*60)
      time = time % (24*60)
      hour = time // 60
      minute = time % 60
      return year, month, day, hour, minute
    
      
  df['year']=df.index.map(lambda x: start_date(x)[0])
  df['month']=df.index.map(lambda x: start_date(x)[1])
  df['day_number']=df.index.map(lambda x: start_date(x)[2])
  df['hour']=df.index.map(lambda x: start_date(x)[3])
  df['minute']=df.index.map(lambda x: start_date(x)[4])

  def get_week_day(index):
      day = int(df.loc[index, 'day_number'])
      month = int(monthdict[df.loc[index, 'month']]+1)
      year = int(df.loc[index, 'year'])
      if day==0:
          # Bug: 179 entries with date 2021-01-0, scripted_day_of_week was indicating 3 in isoweekday
          return 2
      return datetime.date(year, month, day).weekday()

  df['day_of_week']=df.index.map(get_week_day)



  # remove wrong dates (2021/01/00),
  df = df[df['day_number']!=0]
  df = df[df['end_day_number']!=0]
  # drop duplicate columns
  # df.drop("duration", axis=1, inplace=True)

  # Change types
  df['day_number'] = df['day_number'].astype(int)
  df['year'] = df['year'].astype(int)


  ## Export csv

  df.to_csv(f"{zipcode}.csv")





