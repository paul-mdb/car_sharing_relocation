# Analysis

import pandas as pd
import datetime
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import warnings
import gc
matplotlib.use('Agg')
warnings.filterwarnings("ignore")

TIME_BLOCKS = [(0, 5), (6, 9), (10, 11), (12, 13), (14, 15), (16, 18), (19, 23)] # obtained from the daily number of trips in Lisbon
PLOTS_FOLDER = "Plots/"
AVG_FOLDER = "average_availability/"
NO_AVG_FOLDER = "no_average_availability/"
BOOKING_FOLDER = "bookings/"
TABLE_FOLDER = "tables/"

for zipcode in range(75014, 75021):
    print(zipcode)
    df = pd.read_csv(f"{zipcode}.csv", parse_dates=['end_date_time', 'start_date_time', 'kibana_duration_delta_time'])
    df = df[df['kibana_duration']!=0]

    ## UTILS

    def time_block(dataframe, start_time, end_time, scripted_day):
        return dataframe.loc[(dataframe['hour']>= start_time) & (dataframe['hour']<= end_time) & (dataframe['day_of_week']==scripted_day)]

    def time_block_extended(dataframe, start_time, end_time, scripted_day1, scripted_day2):
        return dataframe.loc[(dataframe['hour']>= start_time) & (dataframe['hour']<= end_time) & (dataframe['day_of_week']>=scripted_day1) & (dataframe['day_of_week']<=scripted_day2)]
        

    days = np.sort(df['day_of_week'].unique())
    days_dict = {1: "Monday", 2:"Tuesday", 3:"Wednesday", 4:"Thursday", 5:"Friday", 6: "Saturday", 7: "Sunday"}
    monthdict = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5, "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10,  "Dec": 11}
    inv_month = {v: k for k, v in monthdict.items()}

    df = df.fillna(0)
    
    # Convert floats to ints
    float_col = df.select_dtypes(include=['float64']) # This will select float columns only
    for col in float_col.columns.values:
        df[col] = df[col].astype('int64')

    ## Split overlapping segments to match time blocks and get ALL segments in one block
    def split(index, block_list):
        additional_segments = pd.DataFrame()
        if df.loc[index, 'status']=="FREE":
            start_date_time = df.loc[index, 'start_date_time']
            end_date_time = df.loc[index, 'end_date_time']
            start_hour, end_hour = start_date_time.hour, end_date_time.hour
            start_day, end_day = start_date_time.day, end_date_time.day
            month, end_month = start_date_time.month, end_date_time.month
            year, end_year = start_date_time.year, end_date_time.year

            start_block = 0 # To which block belongs the start hour
            while start_block < len(block_list) and start_hour >= block_list[start_block][0] :
                start_block+=1
            start_block-=1
            end_block = len(block_list)-1 # To which block belongs the end hour
            while end_block >= 0 and end_hour <= block_list[end_block][1] :
                end_block-=1
            end_block+=1

            if start_day==end_day:
                if start_block!=end_block:
                    new_segment = df.loc[index].copy()
                    new_segment['end_date_time']=new_segment['end_date_time'].replace(hour=block_list[start_block][1])
                    additional_segments = additional_segments.append(new_segment, ignore_index = True)
                    for block in range(start_block+1, end_block):
                        new_segment = df.loc[index].copy()
                        new_segment['start_date_time']=new_segment['start_date_time'].replace(hour=block_list[block][0])
                        new_segment['end_date_time']=new_segment['end_date_time'].replace(hour=block_list[block][1])
                        new_segment['kibana_duration_delta_time']=end_date_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime(end_year, end_month, end_day, block_list[block][0], 0).replace(tzinfo=datetime.timezone.utc)
                        new_segment['kibana_duration']=int((end_date_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime(end_year, end_month, end_day, block_list[block][0], 0).replace(tzinfo=datetime.timezone.utc)).seconds/60)
                        additional_segments = additional_segments.append(new_segment, ignore_index = True)
                    new_segment = df.loc[index].copy()
                    new_segment['start_date_time']=new_segment['start_date_time'].replace(hour=block_list[end_block][0])
                    additional_segments = additional_segments.append(new_segment, ignore_index = True)
                    df.drop(index, inplace=True)
            else :
                # Create segments start day
                new_segment = df.loc[index].copy()
                new_segment['end_date_time']=new_segment['end_date_time'].replace(month=month, day=start_day, hour=block_list[start_block][1])
                new_segment['kibana_duration_delta_time']=end_date_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime(year, month, start_day, block_list[start_block][0], 0).replace(tzinfo=datetime.timezone.utc)
                new_segment['kibana_duration']=int((end_date_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime(year, month, start_day, block_list[start_block][0], 0).replace(tzinfo=datetime.timezone.utc)).seconds/60)
                additional_segments = additional_segments.append(new_segment, ignore_index = True)
                for block in range(start_block+1, len(block_list)):
                    new_segment = df.loc[index].copy()
                    new_segment['start_date_time']=new_segment['start_date_time'].replace(hour=block_list[block][0])
                    new_segment['end_date_time']=new_segment['end_date_time'].replace(month=month, day=start_day, hour=block_list[start_block][1])
                    new_segment['kibana_duration_delta_time']=end_date_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime(year, month, start_day, block_list[start_block][0], 0).replace(tzinfo=datetime.timezone.utc)
                    new_segment['kibana_duration']=int((end_date_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime(year, month, start_day, block_list[start_block][0], 0).replace(tzinfo=datetime.timezone.utc)).seconds/60)
                    additional_segments = additional_segments.append(new_segment, ignore_index = True)
                # Create segments end day
                for block in range(0, end_block):
                    new_segment = df.loc[index].copy()
                    new_segment['start_date_time']=new_segment['start_date_time'].replace(month=end_month, day=end_day, hour=block_list[block][0])
                    new_segment['end_date_time']=new_segment['end_date_time'].replace(hour=block_list[start_block][1])
                    new_segment['kibana_duration_delta_time']=end_date_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime(end_year, end_month, end_day, block_list[block][0], 0).replace(tzinfo=datetime.timezone.utc)
                    new_segment['kibana_duration']=int((end_date_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime(end_year, end_month, end_day, block_list[block][0], 0).replace(tzinfo=datetime.timezone.utc)).seconds/60)
                    additional_segments = additional_segments.append(new_segment, ignore_index = True)
                new_segment = df.loc[index].copy()
                new_segment['start_date_time']=new_segment['start_date_time'].replace(month=end_month, day=end_day, hour=block_list[end_block][0])
                new_segment['kibana_duration_delta_time']=end_date_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime(end_year, end_month, end_day, block_list[end_block][0], 0).replace(tzinfo=datetime.timezone.utc)
                new_segment['kibana_duration']=int((end_date_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime(end_year, end_month, end_day, block_list[end_block][0], 0).replace(tzinfo=datetime.timezone.utc)).seconds/60)
                additional_segments = additional_segments.append(new_segment, ignore_index = True)
                # Create segments other days
                try :
                    date = datetime.date(year, month, start_day)
                    date += datetime.timedelta(days=1)
                    end_date = datetime.date(end_year, end_month, end_day)
                except Exception as e:
                    print(e)
                    print(year, month, start_day)
                    return additional_segments
                    
                while date < end_date:
                    for block in range(0, len(block_list)):
                        new_segment = df.loc[index].copy()
                        new_segment['start_date_time']=new_segment['start_date_time'].replace(month=date.month, day=date.day, hour=block_list[block][0])
                        new_segment['end_date_time']=new_segment['end_date_time'].replace(month=date.month, day=date.day, hour=block_list[block][1])
                        new_segment['kibana_duration_delta_time']=end_date_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime(end_year, date.month, date.day, block_list[block][0], 0).replace(tzinfo=datetime.timezone.utc)
                        new_segment['kibana_duration']=int((end_date_time.replace(tzinfo=datetime.timezone.utc) - datetime.datetime(end_year, date.month, date.day, block_list[block][0], 0).replace(tzinfo=datetime.timezone.utc)).seconds/60)
                        additional_segments = additional_segments.append(new_segment, ignore_index = True)
                    date += datetime.timedelta(days=1)
                df.drop(index, inplace=True)
        return additional_segments


    
    split_df = pd.DataFrame()
    for index in df.index:
        split_df = split_df.append(split(index, TIME_BLOCKS), ignore_index = True)
    df = df.append(split_df, ignore_index = True)

    df['hour']=df['start_date_time'].dt.hour
    df['day_of_week']=df['start_date_time'].dt.day_of_week
    df['day_number']=df['start_date_time'].dt.day
    df['month']=df['start_date_time'].dt.month
    df['year']=df['start_date_time'].dt.year
    df = df.sort_values(by="start_date_time", ascending = False)
    df.reset_index(drop=True, inplace = True)
    df.to_csv(f"{zipcode}_split.csv")
    # df = pd.read_csv(f"{zipcode}_split.csv", parse_dates=['end_date_time', 'start_date_time', 'kibana_duration_delta_time'])

    ## PLOTS

    plt.figure(figsize=(25, 15))

    for start_date, end_date in TIME_BLOCKS:
        for day in range(7):

            ## AVERAGE AVAILABILITY

            # Next group is any
            dataframe = time_block(df, start_date, end_date, day).loc[df["status"]=="FREE"]
            dataframe = dataframe[dataframe['kibana_duration']<2000] # filter segments with duration > 2000 minutes = 33 hours
            grouped_df = dataframe.groupby(by=['month', 'day_number', 'year']).agg({'year': 'first', 'month': 'first', 'day_number': 'first', 'kibana_duration': ['mean', 'count', 'std'], 'next_group_id': 'first'})
            durations_means = np.array(grouped_df['kibana_duration']['mean'].values)
            standards = np.array(grouped_df['kibana_duration']['std'].values)
            count = np.array(grouped_df['kibana_duration']['count'].values)
            dates = (grouped_df['year'].astype(str) + "/" + grouped_df['month'].astype(str) + "/" + grouped_df['day_number'].astype(str)).values

            # Next group-id is Client
            dataframe = time_block(df, start_date, end_date, day).loc[df["status"]=="FREE"]
            dataframe = dataframe[dataframe['next_group_id']=='Client']
            dataframe = dataframe[dataframe['kibana_duration']<2000] # filter segments with duration > 2000 minutes = 33 hours
            grouped_df = dataframe.groupby(by=['month', 'day_number', 'year']).agg({'year': 'first', 'month': 'first', 'day_number': 'first', 'kibana_duration': ['mean', 'count', 'std'], 'next_group_id': 'first'})
            client_durations_means = np.array(grouped_df['kibana_duration']['mean'].values)
            client_standards = np.array(grouped_df['kibana_duration']['std'].values)
            client_count = np.array(grouped_df['kibana_duration']['count'].values)
            client_dates = (grouped_df['year'].astype(str) + "/" + grouped_df['month'].astype(str) + "/" + grouped_df['day_number'].astype(str)).values

            # Next group-id is driver
            dataframe = time_block(df, start_date, end_date, day).loc[df["status"]=="FREE"]
            dataframe = dataframe[dataframe['next_group_id']!='Client']
            dataframe = dataframe[dataframe['kibana_duration']<2000] # filter segments with duration > 2000 minutes = 33 hours
            grouped_df = dataframe.groupby(by=['month', 'day_number', 'year']).agg({'year': 'first', 'month': 'first', 'day_number': 'first', 'kibana_duration': ['mean', 'count', 'std'], 'next_group_id': 'first'})
            driver_durations_means = np.array(grouped_df['kibana_duration']['mean'].values)
            driver_standards = np.array(grouped_df['kibana_duration']['std'].values)
            driver_count = np.array(grouped_df['kibana_duration']['count'].values)
            driver_dates = (grouped_df['year'].astype(str) + "/" + grouped_df['month'].astype(str) + "/" + grouped_df['day_number'].astype(str)).values



            x = np.array(count).reshape(-1, 1)
            y = np.array(durations_means)
            a, _, _, _ = np.linalg.lstsq(x, y-1)

            reg = LinearRegression().fit(x, y)
            steep = reg.coef_[0]
            intercept = reg.intercept_
        
            pearson, p_value = pearsonr(count, durations_means)

            
            plt.scatter(client_count,  client_durations_means, c=client_standards, cmap='Greens_r') # Points with lower standard deviation are shaded darker.
            plt.scatter(driver_count,  driver_durations_means, c=driver_standards, cmap='Reds_r') # Points with lower standard deviation are shaded darker.

            # plt.plot(count, a*count+1, c='red', label = "Slope: %a" %(int(a[0]*1000)/1000) + f"\nPearson : {(int(pearson*1000)/1000)}" )
            # plt.plot(count, steep*count+intercept, c='blue')

            """
            for i, v in enumerate(client_count):
                plt.text(client_count[i], 1.01*client_durations_means[i]+3, "%s" %client_dates[i][0], ha="left")
            for i, v in enumerate(driver_count):
                plt.text(driver_count[i], 1.01*driver_durations_means[i]+3, "%s" %driver_dates[i][0], ha="left")
                """

            plt.xlabel("Number of free cars in block")
            plt.ylabel("free duration mean")
            # plt.legend(loc="upper left")
            plt.title(f"{days_dict[day+1]} from {start_date}h to {(end_date+1)%24}h - {zipcode}", fontsize=20)
            plt.savefig(PLOTS_FOLDER+f"{zipcode}/"+AVG_FOLDER+f"AVG_AVAILABILITY-{zipcode}-{days_dict[day+1]}-{start_date}h-{(end_date+1)%24}h.png")
            plt.clf()


            ## NO AVERAGE AVAILABILITY

            # Next group is any
            dataframe = time_block(df, start_date, end_date, day).loc[df["status"]=="FREE"]
            dataframe = dataframe[dataframe['kibana_duration']<2000] # filter segments with duration > 2000 minutes = 33 hours
            grouped_df = dataframe.groupby(by=['month', 'day_number', 'year']).agg({'year': 'first', 'month': 'first', 'day_number': 'first', 'kibana_duration': ['mean', 'count', 'std']})
            count = np.array(grouped_df['kibana_duration']['count'].values)
            dates = np.squeeze((grouped_df['year'].astype(str) + grouped_df['month'].astype(str) + grouped_df['day_number'].astype(str)).values)

            count_dict = dict(list(zip(dates, count)))

            def calculate_count(index):
                date = str(df.loc[index, 'year'])+ str(df.loc[index, 'month']) + str(df.loc[index, 'day_number'])
                return(count_dict[date])

            dataframe['count']=dataframe.index.map(calculate_count)



            count = np.array(dataframe['count'].values)
            durations = np.array(dataframe['kibana_duration'].values)

            grouped_df = dataframe.groupby(by=['count', pd.cut(dataframe['kibana_duration'], 4)]).agg({'count': 'first', 'kibana_duration': ['mean', 'count']})

            # plt.scatter(count,  durations) # Points with lower standard deviation are shaded darker.
            plt.scatter(grouped_df['count']['first'], grouped_df['kibana_duration']['mean'], s=grouped_df['kibana_duration']['count']*2)

            # plt.plot(count, a*count+1, c='red', label = "Slope: %a" %(int(a[0]*1000)/1000) + f"\nPearson : {(int(pearson*1000)/1000)}" )
            # plt.plot(count, steep*count+intercept, c='blue')

            """
            for i, v in enumerate(client_count):
                plt.text(client_count[i], 1.01*client_durations_means[i]+3, "%s" %client_dates[i][0], ha="left")
            for i, v in enumerate(driver_count):
                plt.text(driver_count[i], 1.01*driver_durations_means[i]+3, "%s" %driver_dates[i][0], ha="left")
                """

            plt.xlabel("Number of free cars in block")
            plt.ylabel("free duration mean")
            # plt.legend(loc="upper left")
            plt.title(f"{days_dict[day+1]} from {start_date}h to {(end_date+1)%24}h - {zipcode}", fontsize=20)
            plt.savefig(PLOTS_FOLDER+f"{zipcode}/"+NO_AVG_FOLDER+f"NO_AVG_AVAILABILITY-{zipcode}-{days_dict[day+1]}-{start_date}h-{(end_date+1)%24}h.png")
            plt.clf()


            ## Booking number & estimated demand


            dataframe = time_block(df, start_date, end_date, day)
            df1 = dataframe.groupby(by=['month', 'day_number', 'year'])['status'].apply(lambda x: (x!='FREE').sum()).reset_index(name='booked_count')
            df2 = dataframe.groupby(by=['month', 'day_number', 'year'])['status'].apply(lambda x: (x=='FREE').sum()).reset_index(name='free_count')
            booked_count = np.array(df1['booked_count'].values)
            free_count = df2['free_count'].values
            dates = (df2['year'].astype(int).astype(str) + "/" + df2['month'].astype(int).astype(str) + "/" + df2['day_number'].astype(int).astype(str)).values

            mu = ((1+client_count)/client_durations_means)*(end_date-start_date+1)*60 # queuing theory

            x = np.array(free_count).reshape(-1, 1)
            y = np.array(booked_count)
            a, _, _, _ = np.linalg.lstsq(x, y)
            pearson = np.nan_to_num(pearsonr(free_count, booked_count)[0])


            x = np.array(client_count).reshape(-1, 1)
            y = np.nan_to_num(np.array(mu))
            a_mu, _, _, _ = np.linalg.lstsq(x, y)
            try :
                pearson_mu = np.nan_to_num(pearsonr(client_count, mu)[0])
            except Exception as e:
                print("EXCEPTION")
                print(e)
                print(client_count)
                print(mu)
                pearson_mu = 0

            fig, (ax1, ax2) = plt.subplots(2)

            # fig.rcParams["figure.autolayout"] = True
            # plt.plot(free_count, a*free_count, c='red', label = "Slope: %a" %(int(a[0]*1000)/1000) + f"\nPearson : {(int(pearson*1000)/1000)}" )
            fig.set_figheight(10)
            fig.set_figwidth(25)
            ax1.plot(free_count, booked_count, 'bo', label = "Slope: %a" %(int(a[0]*1000)/1000) + f"\nPearson : {(int(pearson*1000)/1000)}")
            ax1.plot(free_count, free_count, 'k--')
            #ax1.plot(free_count, raw_demand, 'bo', c='red')
            #ax1.plot(free_count, [np.mean(booked_count)]*len(free_count))
            #ax1.plot(free_count, [np.mean(raw_demand+)]*len(free_count), c='red')
            
            try :
                ax2.plot(client_count, mu, 'bo', c='red', label = "Slope: %a" %(int(a_mu[0]*1000)/1000) + f"\nPearson : {(int(pearson_mu*1000)/1000)}")
            except Exception as e :
                print("EXCEPTION")
                print(a_mu)
                print(pearson_mu)
            ax2.plot(free_count, free_count, 'k--')

            for i, v in enumerate(free_count):
                ax1.text(free_count[i], 1.01*booked_count[i]+3, "%s" %dates[i], ha="left")

            #ax1.set_ylim([0,50])
            #ax2.set_ylim([0, 50])
            #ax1.set_xlim([0,60])
            #ax2.set_xlim([0, 60])
            ax2.set_xlabel("Number of free cars in block")
            ax1.set_ylabel("Number of bookings")
            ax2.set_ylabel("Estimated demand")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper left")
            fig.suptitle(f"{days_dict[day+1]} from {start_date}h to {end_date}h - {zipcode}", fontsize=20)
            fig.savefig(PLOTS_FOLDER+f"{zipcode}/"+BOOKING_FOLDER+f"BOOKINGS_DEMAND-{zipcode}-{days_dict[day+1]}-{start_date}h-{(end_date+1)%24}h.png")
            plt.clf()


            ## CORRELATIONS

            cor, cor_days, starts, stops, lengths = [], [], [], [], []
            for day in range(0, 7):
                for u, v in TIME_BLOCKS:
                    dataframe = time_block(df, u, v, day).loc[df["status"]=="FREE"]
                    dataframe = dataframe[dataframe['kibana_duration']<2000] # filter segments with duration > 2000 minutes = 33 hours
                    grouped_df = dataframe.groupby(by=['month', 'day_number', 'year']).agg({'year': 'first', 'month': 'first', 'day_number': 'first', 'kibana_duration': ['mean', 'count', 'std']})
                    durations_means = grouped_df['kibana_duration']['mean'].values
                    count = grouped_df['kibana_duration']['count'].values
                    try :
                        cor.append(pearsonr(count, durations_means)[0])
                        cor_days.append(days_dict[day+1])
                        starts.append(u)
                        stops.append(v)
                        lengths.append(len(count))
                    except Exception as e:
                        print(e)
            cor_df = pd.DataFrame(list(zip(cor_days, starts, stops, cor, lengths)), columns=['day', 'start hour',  'end hour', 'Pearson correlation coefficient', 'Number of blocks'])                    


            ## Tables

            def estimated_demand_mean(df, start_date, end_date, day):
                dataframe = time_block(df, start_date, end_date, day)
                grouped_df = dataframe.groupby(by=['month', 'day_number', 'year']).agg({'year': 'first', 'month': 'first', 'day_number': 'first', 'kibana_duration': ['mean', 'count', 'std']})
                count = np.array(grouped_df['kibana_duration']['count'].values)
                durations_means = np.array(grouped_df['kibana_duration']['mean'].values)
                mu = np.nan_to_num(((1+count)/durations_means)*(end_date-start_date+1)*60) # queuing theory
                return int(np.mean(mu))


            def estimated_demand_std(df, start_date, end_date, day):
                dataframe = time_block(df, start_date, end_date, day)
                grouped_df = dataframe.groupby(by=['month', 'day_number', 'year']).agg({'year': 'first', 'month': 'first', 'day_number': 'first', 'kibana_duration': ['mean', 'count', 'std']})
                count = np.array(grouped_df['kibana_duration']['count'].values)
                durations_means = np.array(grouped_df['kibana_duration']['mean'].values)
                mu = np.nan_to_num(((1+count)/durations_means)*(end_date-start_date+1)*60) # queuing theory
                return (np.std(mu))


            def number_of_bookings_mean(df, start_date, end_date, day):
                dataframe = time_block(df, start_date, end_date, day)
                df1 = dataframe.groupby(by=['month', 'day_number', 'year'])['status'].apply(lambda x: (x!='FREE').sum()).reset_index(name='booked_count')
                booked_count = np.array(df1['booked_count'].values)
                return (np.mean(booked_count))


            def number_of_bookings_std(df, start_date, end_date, day):
                dataframe = time_block(df, start_date, end_date, day)
                df1 = dataframe.groupby(by=['month', 'day_number', 'year'])['status'].apply(lambda x: (x!='FREE').sum()).reset_index(name='booked_count')
                booked_count = np.array(df1['booked_count'].values)
                return (np.std(booked_count))



            features = [estimated_demand_mean, number_of_bookings_mean, estimated_demand_std, number_of_bookings_std]
            feature_names = ['estimated demand mean', 'number of bookings mean', 'estimated_demand_std', 'number_of_bookings_std']


            # Calculating features and storing results in a dedicated dataframe
            array = np.empty(shape=(0, 4+len(features)))
            for day, (u, v) in itertools.product(days, TIME_BLOCKS):
                row = [zipcode, days_dict[day+1], u, v]
                for feature in features :
                    row.append(feature(df, u, v, day))
                array = np.vstack([array, row])

            results = pd.DataFrame(array, columns = ['district', 'day', 'start hour', 'end hour']+feature_names)
            # results.to_csv(PLOTS_FOLDER+f"{zipcode}/"+TABLE_FOLDER+f"RESULTS_TABLE-{zipcode}")

    plt.close()
        # print(gc.collect())