#!/usr/bin/env python3

import time
import math, random
import subprocess
import numpy as np
from datetime import datetime
from dateutil import tz
import pytz
from warnings import warn
from influxdb import InfluxDBClient
import operator
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy
import pandas as pd
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

class NeuroKitWarning(RuntimeWarning):
    """Category for runtime warnings that occur within the NeuroKit library.
    """

def listify(**kwargs):
    """Transforms arguments into lists of the same length.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> nk.listify(a=3, b=[3, 5], c=[3]) #doctest: +ELLIPSIS
    {'a': [3, 3], 'b': [3, 5], 'c': [3, 3]}

    """
    args = kwargs
    maxi = 1

    # Find max length
    for key, value in args.items():
        if isinstance(value, str) is False:
            try:
                if len(value) > maxi:
                    maxi = len(value)
            except TypeError:
                pass

    # Transform to lists
    for key, value in args.items():
        if isinstance(value, list):
            args[key] = _multiply_list(value, maxi)
        else:
            args[key] = _multiply_list([value], maxi)

    return args

def _multiply_list(lst, length):
    q, r = divmod(length, len(lst))
    return q * lst + lst[:r]

def signal_distort(
    signal,
    sampling_rate=1000,
    noise_shape="laplace",
    noise_amplitude=0,
    noise_frequency=100,
    powerline_amplitude=0,
    powerline_frequency=50,
    artifacts_amplitude=0,
    artifacts_frequency=100,
    artifacts_number=5,
    linear_drift=False,
    random_state=None,
    silent=False,
):
    """Signal distortion.

    Add noise of a given frequency, amplitude and shape to a signal.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    noise_shape : str
        The shape of the noise. Can be one of 'laplace' (default) or
        'gaussian'.
    noise_amplitude : float
        The amplitude of the noise (the scale of the random function, relative
        to the standard deviation of the signal).
    noise_frequency : float
        The frequency of the noise (in Hz, i.e., samples/second).
    powerline_amplitude : float
        The amplitude of the powerline noise (relative to the standard
        deviation of the signal).
    powerline_frequency : float
        The frequency of the powerline noise (in Hz, i.e., samples/second).
    artifacts_amplitude : float
        The amplitude of the artifacts (relative to the standard deviation of
        the signal).
    artifacts_frequency : int
        The frequency of the artifacts (in Hz, i.e., samples/second).
    artifacts_number : int
        The number of artifact bursts. The bursts have a random duration
        between 1 and 10% of the signal duration.
    linear_drift : bool
        Whether or not to add linear drift to the signal.
    random_state : int
        Seed for the random number generator. Keep it fixed for reproducible
        results.
    silent : bool
        Whether or not to display warning messages.

    Returns
    -------
    array
        Vector containing the distorted signal.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=10, frequency=0.5)
    >>>
    >>> # Noise
    >>> noise = pd.DataFrame({"Freq100": nk.signal_distort(signal, noise_frequency=200),
    ...                       "Freq50": nk.signal_distort(signal, noise_frequency=50),
    ...                       "Freq10": nk.signal_distort(signal, noise_frequency=10),
    ...                       "Freq5": nk.signal_distort(signal, noise_frequency=5),
    ...                       "Raw": signal}).plot()
    >>> noise #doctest: +SKIP
    >>>
    >>> # Artifacts
    >>> artifacts = pd.DataFrame({"1Hz": nk.signal_distort(signal, noise_amplitude=0,
    ...                                                    artifacts_frequency=1, artifacts_amplitude=0.5),
    ...                           "5Hz": nk.signal_distort(signal, noise_amplitude=0,
    ...                                                    artifacts_frequency=5, artifacts_amplitude=0.2),
    ...                           "Raw": signal}).plot()
    >>> artifacts #doctest: +SKIP

    """
    # Seed the random generator for reproducible results.
    np.random.seed(random_state)

    # Make sure that noise_amplitude is a list.
    if isinstance(noise_amplitude, (int, float)):
        noise_amplitude = [noise_amplitude]

    signal_sd = np.std(signal, ddof=1)
    if signal_sd == 0:
        signal_sd = None

    noise = 0

    # Basic noise.
    if min(noise_amplitude) > 0:
        noise += _signal_distort_noise_multifrequency(
            signal,
            signal_sd=signal_sd,
            sampling_rate=sampling_rate,
            noise_amplitude=noise_amplitude,
            noise_frequency=noise_frequency,
            noise_shape=noise_shape,
            silent=silent,
        )

    # Powerline noise.
    if powerline_amplitude > 0:
        noise += _signal_distort_powerline(
            signal,
            signal_sd=signal_sd,
            sampling_rate=sampling_rate,
            powerline_frequency=powerline_frequency,
            powerline_amplitude=powerline_amplitude,
            silent=silent,
        )

    # Artifacts.
    if artifacts_amplitude > 0:
        noise += _signal_distort_artifacts(
            signal,
            signal_sd=signal_sd,
            sampling_rate=sampling_rate,
            artifacts_frequency=artifacts_frequency,
            artifacts_amplitude=artifacts_amplitude,
            artifacts_number=artifacts_number,
            silent=silent,
        )

    if linear_drift:
        noise += _signal_linear_drift(signal)

    distorted = signal + noise

    return distorted


def _signal_linear_drift(signal):

    n_samples = len(signal)
    linear_drift = np.arange(n_samples) * (1 / n_samples)

    return linear_drift


def _signal_distort_artifacts(
    signal,
    signal_sd=None,
    sampling_rate=1000,
    artifacts_frequency=0,
    artifacts_amplitude=0.1,
    artifacts_number=5,
    artifacts_shape="laplace",
    silent=False,
):

    # Generate artifact burst with random onset and random duration.
    artifacts = _signal_distort_noise(
        len(signal),
        sampling_rate=sampling_rate,
        noise_frequency=artifacts_frequency,
        noise_amplitude=artifacts_amplitude,
        noise_shape=artifacts_shape,
        silent=silent,
    )
    if artifacts.sum() == 0:
        return artifacts

    min_duration = int(np.rint(len(artifacts) * 0.001))
    max_duration = int(np.rint(len(artifacts) * 0.01))
    artifact_durations = np.random.randint(min_duration, max_duration, artifacts_number)

    artifact_onsets = np.random.randint(0, len(artifacts) - max_duration, artifacts_number)
    artifact_offsets = artifact_onsets + artifact_durations

    artifact_idcs = np.array([False] * len(artifacts))
    for i in range(artifacts_number):
        artifact_idcs[artifact_onsets[i] : artifact_offsets[i]] = True

    artifacts[~artifact_idcs] = 0

    # Scale amplitude by the signal's standard deviation.
    if signal_sd is not None:
        artifacts_amplitude *= signal_sd
    artifacts *= artifacts_amplitude

    return artifacts


def _signal_distort_powerline(
    signal, signal_sd=None, sampling_rate=1000, powerline_frequency=50, powerline_amplitude=0.1, silent=False
):

    duration = len(signal) / sampling_rate
    powerline_noise = signal_simulate(
        duration=duration, sampling_rate=sampling_rate, frequency=powerline_frequency, amplitude=1, silent=silent
    )

    if signal_sd is not None:
        powerline_amplitude *= signal_sd
    powerline_noise *= powerline_amplitude

    return powerline_noise


def _signal_distort_noise_multifrequency(
    signal,
    signal_sd=None,
    sampling_rate=1000,
    noise_amplitude=0.1,
    noise_frequency=100,
    noise_shape="laplace",
    silent=False,
):
    base_noise = np.zeros(len(signal))
    params = listify(noise_amplitude=noise_amplitude, noise_frequency=noise_frequency, noise_shape=noise_shape)

    for i in range(len(params["noise_amplitude"])):

        freq = params["noise_frequency"][i]
        amp = params["noise_amplitude"][i]
        shape = params["noise_shape"][i]

        if signal_sd is not None:
            amp *= signal_sd

        # Make some noise!
        _base_noise = _signal_distort_noise(
            len(signal),
            sampling_rate=sampling_rate,
            noise_frequency=freq,
            noise_amplitude=amp,
            noise_shape=shape,
            silent=silent,
        )
        base_noise += _base_noise

    return base_noise


def _signal_distort_noise(
    n_samples, sampling_rate=1000, noise_frequency=100, noise_amplitude=0.1, noise_shape="laplace", silent=False
):

    _noise = np.zeros(n_samples)
    # Apply a very conservative Nyquist criterion in order to ensure
    # sufficiently sampled signals.
    nyquist = sampling_rate * 0.1
    if noise_frequency > nyquist:
        if not silent:
            warn(
                f"Skipping requested noise frequency "
                f" of {noise_frequency} Hz since it cannot be resolved at "
                f" the sampling rate of {sampling_rate} Hz. Please increase "
                f" sampling rate to {noise_frequency * 10} Hz or choose "
                f" frequencies smaller than or equal to {nyquist} Hz.",
                category=NeuroKitWarning
            )
        return _noise
    # Also make sure that at least one period of the frequency can be
    # captured over the duration of the signal.
    duration = n_samples / sampling_rate
    if (1 / noise_frequency) > duration:
        if not silent:
            warn(
                f"Skipping requested noise frequency "
                f" of {noise_frequency} Hz since its period of {1 / noise_frequency} "
                f" seconds exceeds the signal duration of {duration} seconds. "
                f" Please choose noise frequencies larger than "
                f" {1 / duration} Hz or increase the duration of the "
                f" signal above {1 / noise_frequency} seconds.",
                category=NeuroKitWarning
            )
        return _noise

    noise_duration = int(duration * noise_frequency)

    if noise_shape in ["normal", "gaussian"]:
        _noise = np.random.normal(0, noise_amplitude, noise_duration)
    elif noise_shape == "laplace":
        _noise = np.random.laplace(0, noise_amplitude, noise_duration)
    else:
        raise ValueError("NeuroKit error: signal_distort(): 'noise_shape' should be one of 'gaussian' or 'laplace'.")

    if len(_noise) != n_samples:
        _noise = signal_resample(_noise, desired_length=n_samples, method="interpolation")
    return _noise

def signal_resample(
    signal, desired_length=None, sampling_rate=None, desired_sampling_rate=None, method="interpolation"
):
    """Resample a continuous signal to a different length or sampling rate.

    Up- or down-sample a signal. The user can specify either a desired length for the vector, or input
    the original sampling rate and the desired sampling rate.
    See https://github.com/neuropsychology/NeuroKit/scripts/resampling.ipynb for a comparison of the methods.

    Parameters
    ----------
    signal :  Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    desired_length : int
        The desired length of the signal.
    sampling_rate : int
        The original sampling frequency (in Hz, i.e., samples/second).
    desired_sampling_rate : int
        The desired (output) sampling frequency (in Hz, i.e., samples/second).
    method : str
        Can be 'interpolation' (see `scipy.ndimage.zoom()`), 'numpy' for numpy's interpolation
        (see `numpy.interp()`),'pandas' for Pandas' time series resampling, 'poly' (see `scipy.signal.resample_poly()`)
        or 'FFT' (see `scipy.signal.resample()`) for the Fourier method. FFT is the most accurate
        (if the signal is periodic), but becomes exponentially slower as the signal length increases.
        In contrast, 'interpolation' is the fastest, followed by 'numpy', 'poly' and 'pandas'.

    Returns
    -------
    array
        Vector containing resampled signal values.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=100))
    >>>
    >>> # Downsample
    >>> downsampled_interpolation = nk.signal_resample(signal, method="interpolation",
    ...                                                sampling_rate=1000, desired_sampling_rate=500)
    >>> downsampled_fft = nk.signal_resample(signal, method="FFT",
    ...                                      sampling_rate=1000, desired_sampling_rate=500)
    >>> downsampled_poly = nk.signal_resample(signal, method="poly",
    ...                                       sampling_rate=1000, desired_sampling_rate=500)
    >>> downsampled_numpy = nk.signal_resample(signal, method="numpy",
    ...                                        sampling_rate=1000, desired_sampling_rate=500)
    >>> downsampled_pandas = nk.signal_resample(signal, method="pandas",
    ...                                         sampling_rate=1000, desired_sampling_rate=500)
    >>>
    >>> # Upsample
    >>> upsampled_interpolation = nk.signal_resample(downsampled_interpolation,
    ...                                              method="interpolation",
    ...                                              sampling_rate=500, desired_sampling_rate=1000)
    >>> upsampled_fft = nk.signal_resample(downsampled_fft, method="FFT",
    ...                                    sampling_rate=500, desired_sampling_rate=1000)
    >>> upsampled_poly = nk.signal_resample(downsampled_poly, method="poly",
    ...                                     sampling_rate=500, desired_sampling_rate=1000)
    >>> upsampled_numpy = nk.signal_resample(downsampled_numpy, method="numpy",
    ...                                      sampling_rate=500, desired_sampling_rate=1000)
    >>> upsampled_pandas = nk.signal_resample(downsampled_pandas, method="pandas",
    ...                                       sampling_rate=500, desired_sampling_rate=1000)
    >>>
    >>> # Compare with original
    >>> fig = pd.DataFrame({"Original": signal,
    ...                     "Interpolation": upsampled_interpolation,
    ...                     "FFT": upsampled_fft,
    ...                     "Poly": upsampled_poly,
    ...                     "Numpy": upsampled_numpy,
    ...                     "Pandas": upsampled_pandas}).plot(style='.-')
    >>> fig #doctest: +SKIP
    >>>
    >>> # Timing benchmarks
    >>> %timeit nk.signal_resample(signal, method="interpolation",
    ...                            sampling_rate=1000, desired_sampling_rate=500) #doctest: +SKIP
    >>> %timeit nk.signal_resample(signal, method="FFT",
    ...                            sampling_rate=1000, desired_sampling_rate=500) #doctest: +SKIP
    >>> %timeit nk.signal_resample(signal, method="poly",
    ...                            sampling_rate=1000, desired_sampling_rate=500) #doctest: +SKIP
    >>> %timeit nk.signal_resample(signal, method="numpy",
    ...                            sampling_rate=1000, desired_sampling_rate=500) #doctest: +SKIP
    >>> %timeit nk.signal_resample(signal, method="pandas",
    ...                            sampling_rate=1000, desired_sampling_rate=500) #doctest: +SKIP

    See Also
    --------
    scipy.signal.resample_poly, scipy.signal.resample, scipy.ndimage.zoom

    """
    if desired_length is None:
        desired_length = int(np.round(len(signal) * desired_sampling_rate / sampling_rate))

    # Sanity checks
    if len(signal) == desired_length:
        return signal

    # Resample
    if method.lower() == "fft":
        resampled = _resample_fft(signal, desired_length)
    elif method.lower() == "poly":
        resampled = _resample_poly(signal, desired_length)
    elif method.lower() == "numpy":
        resampled = _resample_numpy(signal, desired_length)
    elif method.lower() == "pandas":
        resampled = _resample_pandas(signal, desired_length)
    else:
        resampled = _resample_interpolation(signal, desired_length)

    return resampled


# =============================================================================
# Methods
# =============================================================================


def _resample_numpy(signal, desired_length):
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, desired_length, endpoint=False),  # where to interpolate
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal


def _resample_interpolation(signal, desired_length):
    resampled_signal = scipy.ndimage.zoom(signal, desired_length / len(signal))
    return resampled_signal


def _resample_fft(signal, desired_length):
    resampled_signal = scipy.signal.resample(signal, desired_length)
    return resampled_signal


def _resample_poly(signal, desired_length):
    resampled_signal = scipy.signal.resample_poly(signal, desired_length, len(signal))
    return resampled_signal


def _resample_pandas(signal, desired_length):
    # Convert to Time Series
    index = pd.date_range("20131212", freq="L", periods=len(signal))
    resampled_signal = pd.Series(signal, index=index)

    # Create resampling factor
    resampling_factor = str(np.round(1 / (desired_length / len(signal)), 6)) + "L"

    # Resample
    resampled_signal = resampled_signal.resample(resampling_factor).bfill().values

    # Sanitize
    resampled_signal = _resample_sanitize(resampled_signal, desired_length)

    return resampled_signal


# =============================================================================
# Internals
# =============================================================================


def _resample_sanitize(resampled_signal, desired_length):
    # Adjust extremities
    diff = len(resampled_signal) - desired_length
    if diff < 0:
        resampled_signal = np.concatenate([resampled_signal, np.full(np.abs(diff), resampled_signal[-1])])
    elif diff > 0:
        resampled_signal = resampled_signal[0:desired_length]
    return resampled_signal

def signal_simulate(duration=10, sampling_rate=1000, frequency=1, amplitude=0.5, noise=0, silent=False):
    """Simulate a continuous signal.

    Parameters
    ----------
    duration : float
        Desired length of duration (s).
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    frequency : float or list
        Oscillatory frequency of the signal (in Hz, i.e., oscillations per second).
    amplitude : float or list
        Amplitude of the oscillations.
    noise : float
        Noise level (amplitude of the laplace noise).
    silent : bool
        If False (default), might print warnings if impossible frequencies are queried.

    Returns
    -------
    array
        The simulated signal.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> fig = pd.DataFrame({"1Hz": nk.signal_simulate(duration=5, frequency=1),
    ...                     "2Hz": nk.signal_simulate(duration=5, frequency=2),
    ...                     "Multi": nk.signal_simulate(duration=5, frequency=[0.5, 3], amplitude=[0.5, 0.2])}).plot()
    >>> fig #doctest: +SKIP

    """
    n_samples = int(np.rint(duration * sampling_rate))
    period = 1 / sampling_rate
    seconds = np.arange(n_samples) * period

    signal = np.zeros(seconds.size)
    params = listify(frequency=frequency, amplitude=amplitude)

    for i in range(len(params["frequency"])):

        freq = params["frequency"][i]
        amp = params["amplitude"][i]
        # Apply a very conservative Nyquist criterion in order to ensure
        # sufficiently sampled signals.
        nyquist = sampling_rate * 0.1
        if freq > nyquist:
            if not silent:
                warn(
                    f"Skipping requested frequency"
                    f" of {freq} Hz since it cannot be resolved at the"
                    f" sampling rate of {sampling_rate} Hz. Please increase"
                    f" sampling rate to {freq * 10} Hz or choose frequencies"
                    f" smaller than or equal to {nyquist} Hz.",
                    category=NeuroKitWarning
                )
            continue
        # Also make sure that at leat one period of the frequency can be
        # captured over the duration of the signal.
        if (1 / freq) > duration:
            if not silent:
                warn(
                    f"Skipping requested frequency"
                    f" of {freq} Hz since its period of {1 / freq} seconds"
                    f" exceeds the signal duration of {duration} seconds."
                    f" Please choose frequencies larger than"
                    f" {1 / duration} Hz or increase the duration of the"
                    f" signal above {1 / freq} seconds.",
                    category=NeuroKitWarning
                )
            continue

        signal += _signal_simulate_sinusoidal(x=seconds, frequency=freq, amplitude=amp)
        # Add random noise
        if noise > 0:
            signal += np.random.laplace(0, noise, len(signal))

    return signal


# =============================================================================
# Simple Sinusoidal Model
# =============================================================================
def _signal_simulate_sinusoidal(x, frequency=100, amplitude=0.5):

    signal = amplitude * np.sin(2 * np.pi * frequency * x)

    return signal

def scg_simulate(
    duration=10, length=None, sampling_rate=100, noise=0.01, heart_rate=60, heart_rate_std=1, respiratory_rate=15, systolic=120, diastolic=80, method="simple", random_state=None
):
    """Simulate an scg/EKG signal.

    Generate an artificial (synthetic) scg signal of a given duration and sampling rate using either
    the scgSYN dynamical model (McSharry et al., 2003) or a simpler model based on Daubechies wavelets
    to roughly approximate cardiac cycles.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    length : int
        The desired length of the signal (in samples).
    noise : float
        Noise level (amplitude of the laplace noise).
    heart_rate : int
        Desired simulated heart rate (in beats per minute). The default is 70. Note that for the
        scgSYN method, random fluctuations are to be expected to mimick a real heart rate. These
        fluctuations can cause some slight discrepancies between the requested heart rate and the
        empirical heart rate, especially for shorter signals.
    heart_rate_std : int
        Desired heart rate standard deviation (beats per minute).
    method : str
        The model used to generate the signal. Can be 'simple' for a simulation based on Daubechies
        wavelets that roughly approximates a single cardiac cycle. If 'scgsyn' (default), will use an
        advanced model desbribed `McSharry et al. (2003) <https://physionet.org/content/scgsyn/>`_.
    random_state : int
        Seed for the random number generator.

    Returns
    -------
    array
        Vector containing the scg signal.

    Examples
    ----------
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> scg1 = nk.scg_simulate(duration=10, method="simple")
    >>> scg2 = nk.scg_simulate(duration=10, method="scgsyn")
    >>> pd.DataFrame({"scg_Simple": scg1,
    ...               "scg_Complex": scg2}).plot(subplots=True) #doctest: +ELLIPSIS
    array([<AxesSubplot:>, <AxesSubplot:>], dtype=object)

    See Also
    --------
    rsp_simulate, eda_simulate, ppg_simulate, emg_simulate


    References
    -----------
    - McSharry, P. E., Clifford, G. D., Tarassenko, L., & Smith, L. A. (2003). A dynamical model for
    generating synthetic electrocardiogram signals. IEEE transactions on biomedical engineering, 50(3), 289-294.
    - https://github.com/diarmaidocualain/scg_simulation

    """
    # Seed the random generator for reproducible results
    np.random.seed(random_state)

    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = duration * sampling_rate
    if duration is None:
        duration = length / sampling_rate

    # Run appropriate method

    if method.lower() in ["simple", "daubechies"]:
        # print("method is:", method)
        scg = _scg_simulate_daubechies(
            duration=duration, length=length, sampling_rate=sampling_rate, heart_rate=heart_rate, respiratory_rate=respiratory_rate,  systolic=systolic, diastolic=diastolic
        )
    # else:
    #     # print("method is:", method)
    #     approx_number_beats = int(np.round(duration * (heart_rate / 60)))
    #     scg = _scg_simulate_scgsyn(
    #         sfscg=sampling_rate,
    #         N=approx_number_beats,
    #         Anoise=0,
    #         hrmean=heart_rate,
    #         hrstd=heart_rate_std,
    #         lfhfratio=0.5,
    #         sfint=sampling_rate,
    #         ti=(-70, -15, 0, 15, 100),
    #         ai=(1.2, -5, 30, -7.5, 0.75),
    #         bi=(0.25, 0.1, 0.1, 0.1, 0.4),
    #     )
    #     # Cut to match expected length
    #     scg = scg[0:length]

    # Add random noise
    if noise > 0:
        scg = signal_distort(
            scg,
            sampling_rate=sampling_rate,
            noise_amplitude=noise,
            noise_frequency=[5, 10, 100],
            noise_shape="laplace",
            random_state=random_state,
            silent=True,
        )

    # Reset random seed (so it doesn't affect global)
    np.random.seed(None)
    return scg


# =============================================================================
# Daubechies
# =============================================================================
def _scg_simulate_daubechies(duration=10, length=None, sampling_rate=100, heart_rate=70, respiratory_rate=15, systolic=120, diastolic=80):
    """Generate an artificial (synthetic) scg signal of a given duration and sampling rate.

    It uses a 'Daubechies' wavelet that roughly approximates a single cardiac cycle.
    This function is based on `this script <https://github.com/diarmaidocualain/scg_simulation>`_.

    """
    # The "Daubechies" wavelet is a rough approximation to a real, single, cardiac cycle
    # distance = np.exp((heart_rate - (systolic + diastolic)/2)/10)
    # p = int(round(25/(np.exp(8) - np.exp(-9)) * distance + 9))

    # distance = np.exp((heart_rate - (systolic + diastolic)/2)/10)
    # p = int(round(25/(np.exp(8) - np.exp(-9)) * distance + 9))

    # # print(p)
    # # min_p = 9 max_p = 34
    # cardiac_s = scipy.signal.wavelets.daub(int(p))
    # cardiac_d = scipy.signal.wavelets.daub(int(p)) * (diastolic/systolic)
    # print(f"cardiac_s: {len(cardiac_s)}, cardiac_d: {len(cardiac_d)}")
    
    # cardiac_s = scipy.signal.wavelets.daub(int(systolic/10)) * int(math.sqrt(pow(systolic,2)+pow(heart_rate,2)))
    # # print("cardiac_s:", len(cardiac_s))

    # cardiac_d = scipy.signal.wavelets.daub(int(diastolic/10)) * int(math.sqrt(pow(diastolic,2)+pow(heart_rate,2))*0.3)
    # print("cardiac_d:", len(cardiac_d))

    
    # Add the gap after the pqrst when the heart is resting.
    # cardiac = np.concatenate([cardiac, np.zeros(10)])
    # cardiac = np.concatenate([cardiac_s, cardiac_d])

    cardiac_length = int(100*sampling_rate/heart_rate) #sampling_rate #
    ind = random.randint(17, 34) 
    cardiac_s = scipy.signal.wavelets.daub(ind)
    cardiac_d = scipy.signal.wavelets.daub(ind)*0.3*diastolic/80 # change height to 0.3
    cardiac_s = scipy.signal.resample(cardiac_s, 100)
    cardiac_d = scipy.signal.resample(cardiac_d, 100)
    cardiac_s = cardiac_s[0:40]
    distance = 180-systolic # systolic 81-180
    # distance = cardiac_length - len(cardiac_s) - len(cardiac_d) - systolic # here 140 = 40 (cardiac_s) + 100 (cardiac_d) as below
    zero_signal = np.zeros(distance)
    cardiac = np.concatenate([cardiac_s, zero_signal, cardiac_d])
    # cardiac = scipy.signal.resample(cardiac, 100) # fix every cardiac length to 100
    cardiac = scipy.signal.resample(cardiac, cardiac_length) # fix every cardiac length to 1000/heart_rate

    # Caculate the number of beats in capture time period
    num_heart_beats = int(duration * heart_rate / 60)

    # Concatenate together the number of heart beats needed
    scg = np.tile(cardiac, num_heart_beats)

    # Change amplitude
    # scg = scg * 10
    # scg = scg * 10

    # Resample
    scg = signal_resample(
        scg, sampling_rate=int(len(scg) / 10), desired_length=length, desired_sampling_rate=sampling_rate
    )

    # max_peak = max(scg)
    # peak_threshold = max_peak/s_d + 0.1
    # peaks, _ = scipy.signal.find_peaks(scg, height=peak_threshold)
    
    ### add rr
    num_points = duration * sampling_rate
    x_space = np.linspace(0,1,num_points)
    seg_fre = respiratory_rate / (60/duration)
    seg_amp = max(scg) * 0.00001
    rr_component = seg_amp * np.sin(2 * np.pi * seg_fre * x_space)
    # scg *= rr_component
    # plt.figure(figsize = (16,2))
    # plt.plot(scg)
    # plt.plot(rr_component * 1000)
    # plt.scatter(peaks, scg[peaks], c = 'r')
    
    # #modeified rr component
    # for i in range(len(scg)):
    #     if scg[i] > 0:
    #         scg[i] *= (rr_component[i] + 2 * seg_amp)
    #     elif scg[i] < 0:
    #         scg[i] *= (rr_component[i] + 2 * seg_amp)
    
    scg *= (rr_component + 2 * seg_amp)
    # plt.figure(figsize = (16,2))
    # plt.plot(scg)
    
    # import matplotlib.pyplot as plt
    # # plt.plot(rr_component,'r')
    # plt.plot(scg)
    # plt.show()

    # import pdb; pdb.set_trace()
    return scg