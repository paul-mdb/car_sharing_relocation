## Imports & connection to Elasticsearch
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
import datetime

for zipcode in range(75001, 75021):
  print(zipcode)


  with open('connect.txt') as f:
      str = f.readlines()

  es = Elasticsearch(str)


  ### Change maximum size setting
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
  df = df[['_source.end__date', '_source.car_plate_number', '_source.status', '_source.group_id', '_source.duration', '_source.distance', '_source.location', '_source.end_location', '_source.zipcode', '_source.battery', '_source.end_battery']]

  print("csv fetched from es!")

  ## Utils

  month_duration_dict = {"Jan": 31, "Feb": 28, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30, "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31}

  month_durations = np.array(list(month_duration_dict.values()))
  cum_durations = np.cumsum(month_durations)
  cum_dict = {i+1: cum_durations[i] for i in range(len(cum_durations))}
  cum_dict[0]=0
  week_dict = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
  monthdict = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5, "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10,  "Dec": 11}
  inv_month = {v: k for k, v in monthdict.items()}


  ## Map values and rename columns

  df['_source.group_id'].replace("Zity", "Client", inplace=True) # Map Zity to Client
  df['_source.status'].replace("BOOKED_PARKED", "BOOKED", inplace=True) # Map BOOKED_PARKED to BOOKED (interpolation is performed later)
  df['_source.group_id'].replace("Zity Corporate", "Defleeted", inplace=True) # Map Zity Corporate to Defleeted
  df.rename(columns = {'_source.end__date':'end_date', '_source.car_plate_number':'car_plate_number', '_source.status':'status', '_source.group_id':'group_id', '_source.duration':'kibana_duration', '_source.distance':'distance', '_source.location':'location', '_source.end_location':'end_location', '_source.zipcode':'zipcode'}, inplace = True)


  ## Splitting and creating columns

  df['end_date_time']= pd.to_datetime(df['end_date'], infer_datetime_format=True, utc=True)
  df['delta_battery']=df['_source.end_battery']-df['_source.battery']
  df.drop('_source.end_battery', axis=1, inplace=True)
  df.drop('_source.battery', axis=1, inplace=True)
  df[['latitude', 'longitude']] = df['location'].str.split(',', expand=True)
  df.drop('location', axis=1, inplace=True)
  df[['end_latitude', 'end_longitude']] = df['end_location'].str.split(',', expand=True)
  df.drop('end_location', axis=1, inplace=True)
  df['kibana_duration'] = pd.to_numeric(df['kibana_duration'], errors='coerce')
  df['kibana_duration'] = df['kibana_duration'].replace(np.nan, 0)
  df["kibana_duration"] = df["kibana_duration"].astype(int)
  """
  df[['end_date', 'end_time']] = df['end_date'].str.split('T', expand=True)
  df[['end_time', 'trash']] = df['end_time'].str.split('.', expand=True)
  df.drop('trash', axis=1, inplace=True)
  df[['end_year', 'end_month', "end_day_number"]] = df['end_date'].str.split('-', expand=True)
  df.drop('end_date', axis=1, inplace=True)
  df[['end_hour', 'end_minutes', "end_seconds"]] = df['end_time'].str.split(':', expand=True)
  df.drop('end_time', axis=1, inplace=True)
  df.drop('end_seconds', axis=1, inplace=True)"""


  ### Change types


  df['distance']= pd.to_numeric(df['distance'], errors='coerce')
  df['distance'] = df['distance'].fillna(0).astype(int)
  df['delta_battery']= pd.to_numeric(df['delta_battery'], errors='coerce')
  df['delta_battery'] = df['delta_battery'].fillna(0).astype(int)
  df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
  df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
  df['end_latitude'] = pd.to_numeric(df['end_latitude'], errors='coerce')
  df['end_longitude'] = pd.to_numeric(df['end_longitude'], errors='coerce')
  df['zipcode'] = pd.to_numeric(df['zipcode'], errors='coerce')
  df['kibana_duration'] = pd.to_numeric(df['kibana_duration'], errors='coerce').astype(int)

  ## Sort by plate number and end date

  df = df.sort_values(by=["car_plate_number", "end_date_time"], ascending = False)
  df.reset_index(drop=True, inplace = True)

  ## Remove (BOOKED, Client) when previous segment is (BOOKED, Battery). We do not want to detect when a driver drives back a car from charge (at this moment group id changed from battery to client)

  df['last_group'] = df['group_id'].shift(-1)
  df['last_status'] = df['status'].shift(-1)
  df = df.loc[(df['group_id']!='Client')|(df['status']!="BOOKED")|(df['last_group']!='Battery')|(df['last_status']!="BOOKED")]
  df.drop('last_group', axis=1, inplace=True)
  df.drop('last_status', axis=1, inplace=True)
  df.reset_index(drop=True, inplace = True)



  ## UTILS

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

  ## Next segment group-id column

  df["last_plate"] = df['car_plate_number'].shift(-1)
  df["next_plate"] = df['car_plate_number'].shift(1)
  df["next_group_id"] = df["group_id"].shift(1)
  df.loc[df["next_plate"]!=df["car_plate_number"], 'next_group_id'] = "no next plate"


  ## Keep only "CLIENT" Group-ids

  df = df[df['group_id']=="Client"]

  ## Filter corrupt data

  df = df[df['latitude'].notna()]
  df = df[df['longitude'].notna()]
  df = df[df['end_date_time'].dt.year>2000]
  df = df[df['end_date_time'].dt.day!=0]

  ### 1st INTERPOLATION
    
  df['last_plate'] = df['car_plate_number'].shift(-1)
  df['last_status'] = df['status'].shift(-1)
  df['Status_has_changed'] = df['status']!=df['last_status']
  df.loc[df["last_plate"]!=df["car_plate_number"], 'Status_has_changed'] = True


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


  features_to_change = ['latitude', 'longitude']
  features_to_cumulate = ['distance', 'delta_battery', 'kibana_duration']

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

  df['to_keep'] = df['Status_has_changed'].shift(1)
  df.loc[df["next_plate"]!=df["car_plate_number"], 'to_keep'] = True

  df = df[df["to_keep"]]
  df.drop("to_keep", axis=1, inplace=True)
  df.drop("Status_has_changed", axis=1, inplace=True)

  print("first interpolation done!")

  ### Remove bookings with 0 distance and duration < 10 minutes

  def new_status(index):
      status = df.loc[index, 'status']
      distance = df.loc[index, 'distance']
      location = df.loc[index, 'latitude'], df.loc[index, 'longitude']
      end_location = df.loc[index, 'end_latitude'], df.loc[index, 'end_longitude']
      if status=="BOOKED" and distance==0 and end_location == location:
          return "FREE"
      return status

  df['status']=df.index.map(new_status)

  ### 2nd INTERPOLATION

  df['last_plate'] = df['car_plate_number'].shift(-1)
  df['last_status'] = df['status'].shift(-1)
  df['Status_has_changed'] = df['status']!=df['last_status']
  df.loc[df["last_plate"]!=df["car_plate_number"], 'Status_has_changed'] = True

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

  df['to_keep'] = df['Status_has_changed'].shift(1)
  df.loc[df["next_plate"]!=df["car_plate_number"], 'to_keep'] = True

  df = df[df["to_keep"]]
  df.drop("to_keep", axis=1, inplace=True)
  df.drop("Status_has_changed", axis=1, inplace=True)
  df.drop('last_plate', axis=1, inplace=True)
  df.drop('next_plate', axis=1, inplace=True)
  df.drop('next_group_id', axis=1, inplace=True)
  df.drop('last_status', axis=1, inplace=True)
  df.drop('end_date', axis=1, inplace=True)

  print("second interpolation done!")

  ## Start date column --> Start dates are not reliable in car_history (it may be 1 or 2 hours shifted from the real time because of azure's conversions). Therefore start_date is recalulated from end date and duration of the segment.

  df['kibana_duration_delta_time'] = df['kibana_duration'].apply(pd.to_timedelta, unit = 'm')
  df['start_date_time'] = df['end_date_time'] - df['kibana_duration_delta_time']
  df['day'] = df['start_date_time'].dt.day
  df = df[df['day']!=0] # remove wrong dates (2021/01/00)
  df.drop('day', axis=1, inplace=True)
  df['day_of_week'] = df['start_date_time'].dt.day_of_week


  ## Export csv

  df = df.sort_values(by="start_date_time", ascending = False)
  df.reset_index(drop=True, inplace = True)
  df.to_csv(f"{zipcode}.csv")





