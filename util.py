#!/usr/bin/env python3

import time
import math, random
import subprocess
import numpy as np
from datetime import datetime
from dateutil import tz
import pytz

from influxdb import InfluxDBClient
import operator
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# This function converts the time string to epoch time xxx.xxx (second.ms).
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def local_time_epoch(time, zone):
    local_tz = pytz.timezone(zone)
    localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
    local_dt = local_tz.localize(localTime, is_dst=None)
    # utc_dt = local_dt.astimezone(pytz.utc)
    epoch = local_dt.timestamp()
    # print("epoch time:", epoch) # this is the epoch time in seconds, times 1000 will become epoch time in milliseconds
    # print(type(epoch)) # float
    return epoch 

# This function converts the epoch time xxx.xxx (second.ms) to time string.
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def epoch_time_local(epoch, zone):
    local_tz = pytz.timezone(zone)
    time = datetime.fromtimestamp(epoch).astimezone(local_tz).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return time 

# This function converts the grafana URL time to epoch time. For exmaple, given below URL
# https://sensorweb.us:3000/grafana/d/OSjxFKvGk/caretaker-vital-signs?orgId=1&var-mac=b8:27:eb:6c:6e:22&from=1612293741993&to=1612294445244
# 1612293741993 means epoch time 1612293741.993; 1612294445244 means epoch time 1612294445.244
def grafana_time_epoch(time):
    return time/1000

# This function write an array of data to influxdb. It assumes the sample interval is 1/fs.
# influx - the InfluxDB info including ip, db, user, pass. Example influx = {'ip': 'https://sensorweb.us', 'db': 'algtest', 'user':'test', 'passw':'sensorweb'}
# dataname - the dataname such as temperature, heartrate, etc
# timestamp - the epoch time (in second) of the first element in the data array, such as datetime.now().timestamp()
# fs - the sampling frequency of readings in data
# unit - the unit location name tag
def write_influx(influx, unit, table_name, data_name, data, start_timestamp, fs):
    # print("epoch time:", timestamp) 
    timestamp = start_timestamp
    max_size = 100
    count = 0
    total = len(data)
    prefix_post  = "curl -i -k -XPOST \'"+ influx['ip']+":8086/write?db="+influx['db']+"\' -u "+ influx['user']+":"+ influx['passw']+" --data-binary \' "
    http_post = prefix_post
    for value in data:
        count += 1
        http_post += "\n" + table_name +",location=" + unit + " "
        http_post += data_name + "=" + str(value) + " " + str(int(timestamp*10e8))
        timestamp +=  1/fs
        if(count >= max_size):
            http_post += "\'  &"
            # print(http_post)
            print("Write to influx: ", table_name, data_name, count)
            subprocess.call(http_post, shell=True)
            total = total - count
            count = 0
            http_post = prefix_post
    if count != 0:
        http_post += "\'  &"
        # print(http_post)
        print("Write to influx: ", table_name, data_name, count, data)
        subprocess.call(http_post, shell=True)

# This function read an array of data from influxdb.
# influx - the InfluxDB info including ip, db, user, pass. Example influx = {'ip': 'https://sensorweb.us', 'db': 'testdb', 'user':'test', 'passw':'sensorweb'}
# dataname - the dataname such as temperature, heartrate, etc
# start_timestamp, end_timestamp - the epoch time (in second) of the first element in the data array, such as datetime.now().timestamp()
# unit - the unit location name tag
def read_influx(influx, unit, table_name, data_name, start_timestamp, end_timestamp, condition="location"):
    client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=True)
    query = 'SELECT "' + data_name + '" FROM "' + table_name + f'" WHERE "{condition}" = \''+unit+'\' AND time >= '+ str(int(start_timestamp*10e8))+' AND time <= '+str(int(end_timestamp*10e8))
    # query = 'SELECT last("H") FROM "labelled" WHERE ("location" = \''+unit+'\')'

    # print(query)
    result = client.query(query)
    # print(result)

    points = list(result.get_points())
    values =  list(map(operator.itemgetter(data_name), points))
    times  =  list(map(operator.itemgetter('time'),  points))
    # print(times)
    # times = [local_time_epoch(item[:-1], "UTC") for item in times] # convert string time to epoch time
    # print(times)

    data = values #np.array(values)
    # print(data, times)
    return data, times

def generate_signal(length_seconds, sampling_rate, frequencies, func="sin", add_noise=0, plot=False):
    r"""
    Generate a n-D array, `length_seconds` seconds signal at `sampling_rate` sampling rate.
    Cited from https://towardsdatascience.com/hands-on-signal-processing-with-python-9bda8aad39de
    
    Args:
        length_seconds : float
            Duration of signal in seconds (i.e. `10` for a 10-seconds signal, `3.5` for a 3.5-seconds signal)
        sampling_rate : int
            The sampling rate of the signal.
        frequencies : 1 or 2 dimension python list a floats
            An array of floats, where each float is the desired frequencies to generate (i.e. [5, 12, 15] to generate a signal containing a 5-Hz, 12-Hz and 15-Hz)
            2 dimension python list, i.e. [[5, 12, 15],[1]], to generate a signal with 2 channels, where the second channel containing 1-Hz signal
        func : string, optional, default: sin
            The periodic function to generate signal, either `sin` or `cos`
        add_noise : float, optional, default: 0
            Add random noise to the signal, where `0` has no noise
        plot : boolean, optional, default: False
            Plot the generated signal
    
    Returns:
        signal : n-d ndarray
            Generated signal, a numpy array of length `sampling_rate*length_seconds`
    Usage:
        >>> s = generate_signal(length_seconds=4, 
        >>>     sampling_rate=100, 
        >>>     frequencies=[2], 
        >>>     plot=True
        >>> )
        >>> 
        >>> s = generate_signal(length_seconds=4, 
        >>>     sampling_rate=100, 
        >>>     frequencies=[1,2], 
        >>>     func="cos", 
        >>>     add_noise=0.5, 
        >>>     plot=True
        >>> )
        >>> 
        >>> s = generate_signal(length_seconds=3.5, 
        >>>     sampling_rate=100, 
        >>>     frequencies=[[1,2],[1],[2]],  
        >>>     plot=True
        >>> )
    """
    
    frequencies = np.array(frequencies, dtype=object)
    assert len(frequencies.shape) == 1 or len(frequencies.shape) == 2, "frequencies must be 1d or 2d python list"
    
    expanded = False
    if isinstance(frequencies[0], int):
        frequencies = np.expand_dims(frequencies, axis=0)
        expanded = True
    
    sampling_rate = int(sampling_rate)
    npnts = int(sampling_rate*length_seconds)  # number of time samples
    time = np.arange(0, npnts)/sampling_rate
    signal = np.zeros((frequencies.shape[0],npnts))
    
    for channel in range(0,frequencies.shape[0]):
        for fi in frequencies[channel]:
            if func == "cos":
                signal[channel] = signal[channel] + np.cos(2*np.pi*fi*time)
            else:
                signal[channel] = signal[channel] + np.sin(2*np.pi*fi*time)
    
        # normalize
        max = np.repeat(signal[channel].max()[np.newaxis], npnts)
        min = np.repeat(signal[channel].min()[np.newaxis], npnts)
        signal[channel] = (2*(signal[channel]-min)/(max-min))-1
    
    if add_noise:        
        noise = np.random.uniform(low=0, high=add_noise, size=(frequencies.shape[0],npnts))
        signal = signal + noise

    if plot:
        plt.plot(time, signal.T)
        plt.title('Signal with sampling rate of '+str(sampling_rate)+', lasting '+str(length_seconds)+'-seconds')
        plt.xlabel('Time (sec.)')
        plt.ylabel('Amplitude')
        plt.show()
    
    if expanded:
        signal = signal[0]
        
    return signal

def read_heartbeat_sample():
    unit = "b8:27:eb:7d:dd:a4"
    table_name = "Z"
    data_name = "value"
    start_timestamp = local_time_epoch("2023-01-08T23:45:33.000", "America/New_York")
    end_timestamp = local_time_epoch("2023-01-08T23:45:43.000", "America/New_York")
    influx = {'ip': 'https://homedots.us', 'db': 'shake', 'user':'test', 'passw':'sensorweb'}

    data, time = read_influx(influx, unit, table_name, data_name, start_timestamp, end_timestamp)
    if(len(data) == 0):
        print("No data in the chosen time range!")
        quit()
    return data, time

def read_footstep_sample():
    unit = "b8:27:eb:7d:dd:a4"
    table_name = "Z"
    data_name = "value"
    start_timestamp = local_time_epoch("2023-01-14T20:27:03.000", "America/New_York")
    end_timestamp = local_time_epoch("2023-01-14T20:27:13.000", "America/New_York")
    influx = {'ip': 'https://homedots.us', 'db': 'shake', 'user':'test', 'passw':'sensorweb'}

    data, time = read_influx(influx, unit, table_name, data_name, start_timestamp, end_timestamp)
    if(len(data) == 0):
        print("No data in the chosen time range!")
        quit()
    return data, time

if __name__ == '__main__':
    import datasim as nk

    data = generate_signal(10, 100, [1,3])
    plt.figure(figsize=(20, 4))
    plt.title("simulated sine waves")
    # plt.plot (data1)
    # plt.plot (data2)
    plt.plot (data)

    fs = 100
    duration = 10 # 10 seconds
    noise = 0.2
    heart_rate = 60 # random.randint(50, 150)
    respiratory_rate = 15 # random.randint(10, 30)
    systolic = random.randint(100, 160)
    diastolic = random.randint(60,100) #+ systolic
    # print('hr:', heart_rate, 'rr:', respiratory_rate, 
    #     'sp:', systolic, 'dp:', diastolic)
    data = nk.scg_simulate(duration=duration, sampling_rate=fs, noise=noise, heart_rate=heart_rate, respiratory_rate=respiratory_rate, systolic=systolic, diastolic=diastolic)
    plt.figure(figsize=(20, 4))
    plt.title("simulated vital signs")
    plt.plot (data)

    data, time = read_heartbeat_sample()
    plt.figure(figsize=(20, 4))
    plt.title("heartbeat induced vibrations")
    plt.plot(data)

    data, time = read_footstep_sample()
    plt.figure(figsize=(20, 4))
    plt.title("footstep induced vibrations")
    plt.plot(data)

    plt.show()
