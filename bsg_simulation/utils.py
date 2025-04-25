"""
Created on Sep 11 2024
@author: Yida Zhang
"""

import warnings
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy
from scipy.signal import find_peaks, resample
from scipy.interpolate import interp1d, PchipInterpolator
import pywt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
from scipy.signal import butter, filtfilt

# [hour]+[heart_rate]+[respiratory_rate]+[MAP/TPR/heart_rate]+[TPR]+[MAP]

def get_tpr_from_signal(signal):
    second_peak = signal[find_peaks(signal)[0][1]]
    # print(second_peak,"==")
    tmp = (second_peak - 3.679) / (5.977 - 3.679) - 1
    return map_range(tmp, original_min = -0.4, original_max = 0.8, new_min = 0.002, new_max = 0.006)


def butter_bandstop(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs 
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def white_noise(dataset, mag, low_frequency, high_frequency):
    for i in range(dataset.shape[0]):
        noise = mag * np.random.uniform(-1,1,1000)
        b, a = butter_bandpass(low_frequency, high_frequency, 100, order=5)
        high_freq_noise = filtfilt(b, a, noise)
        dataset[i,:1000] = dataset[i,:1000] + high_freq_noise
    return dataset


def denoise(dataset, low_frequency, high_frequency):
    b, a = butter_bandstop(low_frequency, high_frequency, 100, order=5)
    for i in range(dataset.shape[0]):
        dataset[i,:1000]= filtfilt(b, a, dataset[i,:1000])
    return dataset

def plot_fft(signal):
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result) / 1000
    frequencies = np.fft.fftfreq(1000, 1/100)

    half_length = 1000 // 2
    frequencies = frequencies[:half_length]
    fft_magnitude = fft_magnitude[:half_length]

    plt.figure(figsize=(8, 5))
    plt.plot(frequencies, fft_magnitude)
    plt.title('Frequency Domain of the Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def add_respiration(dataset, seg_amp = 0.5):
    num_points = 10 * 100
    x_space = np.linspace(0,1,num_points)
    for i in range(dataset.shape[0]):
        rr = random.randint(12,20)
        seg_fre = rr / (60 / 10)
        seg_amp = seg_amp
        rr_component = seg_amp * np.sin(2 * np.pi * seg_fre * x_space)
        dataset[i,:1000] = dataset[i,:1000] * (rr_component + 1)
        dataset[i,-4] = rr

    return dataset

def map_num_point_to_area(sig, num_point, num_heart_beats, sampling_rate, duration):
    cardiac = change_width(sig, num_point)
    signal = np.tile(cardiac, num_heart_beats)

    x_original = np.linspace(0, len(signal)-1, len(signal))
    interpolator = PchipInterpolator(x_original, signal)
    x_resampled = np.linspace(x_original.min(), x_original.max(), sampling_rate * duration)
    signal = interpolator(x_resampled)
    area_now = calculate_area(signal)
    return area_now

def find_best_integer_input(target_value, sig, num_heart_beats, sampling_rate, duration, lower_bound=-10, upper_bound=20):
    lower = int(lower_bound)
    upper = int(upper_bound)

    while lower <= upper:
        mid = (lower + upper) // 2 
        mapped_mid = map_num_point_to_area(sig, mid, num_heart_beats, sampling_rate, duration)

        if mapped_mid == target_value:
            return mid 

        if mapped_mid < target_value:
            lower = mid + 1
        else:
            upper = mid - 1

    candidates = []
    for i in range(max(lower_bound, upper - 1), min(upper_bound, lower + 2)):
        candidates.append(i)
    closest_input = min(candidates, key=lambda x: abs(map_num_point_to_area(sig, x, num_heart_beats, sampling_rate, duration) - target_value))
    return closest_input



def change_width(signal, num_added_point):
    notch = find_peaks(-signal)[0][0]
    notch_parallel = 52
    seg1 = signal[:notch]
    seg2 = signal[notch:notch_parallel]
    seg3 = signal[notch_parallel:]
    new_seg1 = interp1d(np.linspace(0, 1, len(seg1)), seg1, kind="linear")(np.linspace(0, 1, len(seg1) + num_added_point))
    new_seg3 = interp1d(np.linspace(0, 1, len(seg3)), seg3, kind="linear")(np.linspace(0, 1, len(seg3) - num_added_point))
    new_signal = np.concatenate([new_seg1, seg2, new_seg3])
    return new_signal


def sine_wave_value(x, amp):
    amplitude = amp 
    offset = 1   
    frequency = 2 * np.pi / 24  
    
    y = amplitude * np.sin(frequency * x) + offset
    return y

def calculate_area(signal):
    
    signal_inverted =  - signal
    peaks, _ = find_peaks(signal_inverted[10:])
    peaks += 10
    point_start = 0
    point_end = peaks[0]
    
    area = 0
    for i in range(point_start,point_end+1):
        area += signal[i]
    return area

def map_range(TPR, original_min = 0.002, original_max = 0.006, new_min = -0.4, new_max = 0.8):
    return new_min + (TPR - original_min) * (new_max - new_min) / (original_max - original_min)

def change_notch_height(TPR):
    # max = 0.002, min = 0.006
    tmp = map_range(TPR)
    # print(tmp)
    signal = np.load("./data/template_ABP.npy")
    signal[35:53] = (signal[35:53] - 3.68) * (1 + tmp) + 3.68
    return signal

## Generate signal   
def signal_simulate(**kwargs):
    args = {
        'num_rows' : 1,
        'duration' : 10, 
        'sampling_rate' : 100,
        'heart_rate' : (50,100),
        'MAP' : (50,100),
        'TPR' : (0.002, 0.006),
        'random_state' : None,
        'silent' : False,
        'data_file' : "./data.npy"
    }

    args.update(kwargs)
    simulated_data = []
    ind = 0
    # for ind in tqdm(range(args['num_rows'])):
    while ind < args['num_rows']:
        # if ind % 1000 == 0:
            # print(ind)
        heart_rate = random.randint(args['heart_rate'][0], args['heart_rate'][1])
        respiratory_rate = 0
        MAP = random.randint(args['MAP'][0], args['MAP'][1])
        TPR = random.uniform(args['TPR'][0], args['TPR'][1])
        hour = 0
       
        data = _signal_simulate(
            duration = args['duration'], 
            sampling_rate = args['sampling_rate'], 
            heart_rate = heart_rate,  
            MAP = MAP, 
            TPR = TPR,
            random_state = args['random_state'],
            silent = args['silent']
        )
        if isinstance(data, bool):
            continue
        else:
            ind += 1
            # print(ind)
        simulated_data.append(list(data)+[hour]+[heart_rate]+[respiratory_rate]+[MAP/TPR/heart_rate]+[TPR]+[MAP])

    simulated_data = np.asarray(simulated_data)
    if args['num_rows'] == 1:
        return simulated_data.flatten()
    else:
        np.save(args['data_file'], simulated_data)
        # print(f"{args['data_file']} is generated and saved!")

def _signal_simulate(**kwargs):
    args = {
        'duration' : 10, 
        'sampling_rate' : 100, 
        'heart_rate' : 70, 
        'MAP' : 75, 
        'TPR' : 0.004,
        'random_state' : None,
        'silent' : False
    }
    args.update(kwargs)

    # Seed the random generator for reproducible results
    np.random.seed(args['random_state'])
    # Caculate the number of beats in capture time period
    num_heart_beats = int(args['duration'] * args['heart_rate'] / 60)
    
    # generate a random heartbeat cycle
    # ============================================
    template = np.load("./data/template_ABP.npy")
    template_new = change_notch_height(args['TPR'])
    needed_area = args['MAP'] / args['heart_rate'] / args['TPR']
    if (needed_area > map_num_point_to_area(template_new, 20, num_heart_beats, args['sampling_rate'], args['duration'])) or (needed_area < map_num_point_to_area(template_new, -10, num_heart_beats, args['sampling_rate'], args['duration'])):
        return False
    best_integer_input = find_best_integer_input(needed_area, template_new, num_heart_beats, args["sampling_rate"], args['duration'])
    template = change_width(template_new, best_integer_input)
    cardiac = template
    # ============================================
    
    signal = np.tile(cardiac, num_heart_beats)
    x_original = np.linspace(0, len(signal)-1, len(signal))
    interpolator = PchipInterpolator(x_original, signal)
    x_resampled = np.linspace(x_original.min(), x_original.max(), args['sampling_rate'] * args['duration'])
    signal = interpolator(x_resampled)

    # Reset random seed (so it doesn't affect global)
    np.random.seed(None)
    return signal

def even_separate(redundant_point, n):
    list_tmp = [0] * n
    for i in range(redundant_point):
        list_tmp[i % n] += 1
    return list_tmp

def generate_BSG_from_ABP(abp):
    sig = abp
    sig_inverted = -sig

    # Get all the key points
    peaks1, _ = find_peaks(sig)
    peaks2, _ = find_peaks(-sig)
    notch = peaks2[::2]
    peaks = peaks1[::2]
    second_peaks = peaks1[1::2]

    peaks_to_second_peak_timestamp = []
    not_important_timestamp = []
    for i in range(len(peaks)):

        # Get the key point between peak and notch
        interval_length = 9
        # print(interval_length)
        num_interval = (notch[i] - peaks[i]) // interval_length
        redundant_point = (notch[i] - peaks[i]) % interval_length

        # Prevent the number of intervals in each heartbeat from being different.
        if i != 0:
            if num_interval != len(peaks_to_second_peak_timestamp[-1]) - 2:
                num_interval = len(peaks_to_second_peak_timestamp[-1]) - 2
                redundant_point = (notch[i] - peaks[i]) % num_interval
                interval_length = (notch[i] - peaks[i]) // num_interval
        x_timestamp = [peaks[i]]
        list_tmp = even_separate(redundant_point, num_interval)
        for j in range(len(list_tmp)):
            x_timestamp.append(x_timestamp[-1] + interval_length + list_tmp[j])
        # print(x_timestamp, second_peaks[i], notch[i])
        # if not(x_timestamp[-1] + interval_length == second_peaks[i]):
        #     raise ValueError("x_timestamp[-1] + num_interval == second_peaks[i]")
        x_timestamp.append(second_peaks[i])

        exist_num_interval = len(x_timestamp) - 1
        peaks_to_second_peak_timestamp.append(x_timestamp)
        # print(x_timestamp)

        # not_important_timestamp is one less than peaks_to_notch_timestamp
        if i == len(peaks) - 1:
            continue
        start_point = second_peaks[i]
        end_point = peaks[i + 1]

        odd_even = exist_num_interval % 2
        num_interval = (end_point - start_point) // interval_length
        if not(num_interval % 2 == odd_even):
            num_interval = num_interval - 1

        redundant_point = (end_point - start_point) - num_interval * interval_length

        if i != 0:
            if num_interval != len(not_important_timestamp[-1]) + 1:
                num_interval = len(not_important_timestamp[-1]) + 1
                redundant_point = (end_point - start_point) % num_interval
                interval_length = (end_point - start_point) // num_interval

        list_tmp = even_separate(redundant_point, num_interval)
        x_timestamp = [second_peaks[i] + interval_length + list_tmp[0]]
        for j in range(1, len(list_tmp)):
            x_timestamp.append(x_timestamp[-1] + interval_length + list_tmp[j])
        if x_timestamp[-1] != peaks[i + 1]:
            raise ValueError("sdfsdf")
        x_timestamp.pop()

        not_important_timestamp.append(x_timestamp)

    # for i in range(len(peaks_to_notch_timestamp)):
    #     peaks_to_notch_timestamp[i] = list(peaks_to_notch_timestamp[i])

    total_timestamp = []
    # for j in range(len(peaks_to_second_peak_timestamp)):
    #     print(peaks_to_second_peak_timestamp[j])
    for i in range(len(not_important_timestamp)):
        total_timestamp.append(peaks_to_second_peak_timestamp[i])
        total_timestamp.append(not_important_timestamp[i])
    total_timestamp.append(peaks_to_second_peak_timestamp[-1])
    total_timestamp = [item for sublist in total_timestamp for item in sublist]

    # Start region
    timestamp_tmp = [0]
    redundant_point = peaks[0] % interval_length
    n = peaks[0] // interval_length
    if n % 2 == 0:
        top_start = True
    else:
        top_start = False
    if n == 0:
        top_start = False
        timestamp_tmp = [0]
    else:
        list_tmp = even_separate(redundant_point, n)
        for i in range(len(list_tmp)):
            timestamp_tmp.append(timestamp_tmp[-1] + interval_length + list_tmp[i])
        if timestamp_tmp[-1] != peaks[0]:
            raise ValueError("timestamp_tmp[-1] != peaks[0]")
        timestamp_tmp.pop()
    total_timestamp = timestamp_tmp + total_timestamp

    # End region
    timestamp_tmp = [second_peaks[-1]]
    redundant_point = (999 - second_peaks[-1]) % interval_length
    list_tmp = even_separate(redundant_point, (999 - second_peaks[-1]) // interval_length)
    for i in range(0, len(list_tmp)):
        timestamp_tmp.append(timestamp_tmp[-1] + interval_length + list_tmp[i])
    timestamp_tmp.reverse()
    timestamp_tmp.pop()
    timestamp_tmp.reverse()
    total_timestamp = total_timestamp + timestamp_tmp
    if timestamp_tmp[-1] != 999:
        raise ValueError("timestamp_tmp[-1] != 999")

    waveform = []
    if top_start:
        for i in range(len(total_timestamp)):
            if i % 2 == 0:
                waveform.append(sig[total_timestamp[i]])
            else:
                waveform.append(sig_inverted[total_timestamp[i]])
    else:
        for i in range(len(total_timestamp)):
            if i % 2 == 0:
                waveform.append(sig_inverted[total_timestamp[i]])
            else:
                waveform.append(sig[total_timestamp[i]])

    key_points = np.concatenate([peaks, second_peaks, notch])
    new_total_timestamp = []
    new_waveform = []
    for i in range(len(total_timestamp)):
        if total_timestamp[i] in key_points:
            new_total_timestamp += [total_timestamp[i] - 0.3, total_timestamp[i], total_timestamp[i] + 0.3]
            new_waveform += [waveform[i], waveform[i], waveform[i]]
        else:
            new_total_timestamp.append(total_timestamp[i])
            new_waveform.append(waveform[i])
    new_total_timestamp.insert(1, (new_total_timestamp[0] + new_total_timestamp[1]) / 2)
    new_waveform.insert(1, (new_waveform[0] + new_waveform[1]) / 2)
    # print(new_total_timestamp)
    interpolation_function = interp1d(new_total_timestamp, new_waveform, kind='cubic')
    x_new = range(1000)
    y_resampled = interpolation_function(x_new)
    return y_resampled