import warnings
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy
from scipy.signal import find_peaks, resample, filtfilt, butter
from scipy.interpolate import interp1d
import pywt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

def change_width(signal, num_added_point):
    length = len(signal.flatten())
    x = np.linspace(0, 1, length)
    start_idx = 0
    end_idx = 37

    interp_x = np.linspace(x[start_idx], x[end_idx-1], end_idx - start_idx + num_added_point) 
    interp_signal = interp1d(x[start_idx:end_idx], signal[start_idx:end_idx], kind='linear')(interp_x)


    new_signal = np.concatenate((signal[:start_idx], interp_signal, signal[end_idx:]))
    new_signal = resample(new_signal,100)
    return new_signal


def get_num_added_point_from_TPR(TPR):
    alpha= 0.2
    x_min, x_max = 0.005, 0.015
    y_min, y_max = 10, 80


    y = y_min + (y_max - y_min) * ((np.log(TPR) - np.log(x_min)) / (np.log(x_max) - np.log(x_min))) ** alpha
    
    return y


def sine_wave_value(x, amp):
    amplitude = amp 
    offset = 1 
    frequency = 2 * np.pi / 24 
    
    y = amplitude * np.sin(frequency * x) + offset
    return y

def calculate_area(signal):
    
    signal_inverted =  - signal
    peaks, _ = find_peaks(signal_inverted)
    point_start = 0
    point_end = peaks[1]
    
    area = 0
    for i in range(point_start,point_end+1):
        area += signal[i]
    return area

## Generate signal   
def signal_simulate(**kwargs):
    args = {
        'num_rows' : 1,
        'duration' : 10, 
        'sampling_rate' : 100,
        'heart_rate' : (50,150),
        'respiratory_rate' : (10,30),
        'MAP' : (50,100),
        'TPR' : 0.01,
        'TPR_noise' : 0.001,
        'change_with_time' : False,
        'change_with_time_amp' : 0.1,
        'random_state' : None,
        'silent' : False,
        'data_file' : "./data.npy"
    }

    args.update(kwargs)
    simulated_data = []

    for ind in tqdm(range(args['num_rows'])):
        heart_rate = random.randint(args['heart_rate'][0], args['heart_rate'][1])
        respiratory_rate = 0
        MAP = random.randint(args['MAP'][0], args['MAP'][1])
        if args['change_with_time'] == False:
            TPR = args['TPR'] + random.uniform(-args['TPR_noise'], args['TPR_noise'])
            hour = 0
        else:
            hour = random.uniform(0,24)
            TPR = args['TPR'] * sine_wave_value(hour, args['change_with_time_amp']) + random.uniform(-args['TPR_noise'], args['TPR_noise'])
       
        data = _signal_simulate(
            duration = args['duration'], 
            sampling_rate = args['sampling_rate'], 
            heart_rate = heart_rate,  
            add_respiratory = args['add_respiratory'],
            respiratory_rate = respiratory_rate, 
            MAP = MAP, 
            TPR = TPR,
            random_state = args['random_state'],
            silent = args['silent']
        )
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
        'add_respiratory': True,
        'respiratory_rate' : 20, 
        'MAP' : 75, 
        'TPR' : 0.01,
        'random_state' : None,
        'silent' : False
    }

    args.update(kwargs)

    # Seed the random generator for reproducible results
    np.random.seed(args['random_state'])
    cardiac_length = int(100 * args['sampling_rate'] / args['heart_rate']) 
    
    # ============================================
    template = np.load("./data/template_SV_new.npy")
    num_added_point = get_num_added_point_from_TPR(args['TPR'])
    template = change_width(template, int(num_added_point))
    cardiac = template
    # cardiac = resample(template, cardiac_length) # fix every cardiac length to 1000/heart_rate
    # ============================================
    
    # Caculate the number of beats in capture time period
    num_heart_beats = int(args['duration'] * args['heart_rate'] / 60)

    # Concatenate together the number of heart beats needed
    signal = np.tile(cardiac, num_heart_beats)

    # Resample
    signal = signal_resample(
        signal, 
        sampling_rate = int(len(signal) / 10),
        desired_length = args['sampling_rate'] * args['duration'],
        desired_sampling_rate = args['sampling_rate']
    )

    area = calculate_area(signal)
    # print(args['MAP'], args['heart_rate'], args['TPR'])
    needed_area = args['MAP'] / args['heart_rate'] / args['TPR']
    signal = (needed_area / area) * signal

    ### add rr
    if args['add_respiratory']:
        num_points = args['duration'] * args['sampling_rate']
        x_space = np.linspace(0,1,num_points)
        seg_fre = args['respiratory_rate'] / (60 / args['duration'])
        seg_amp = max(signal) * 0.00001
        rr_component = seg_amp * np.sin(2 * np.pi * seg_fre * x_space)
        signal *= (rr_component + 2 * seg_amp)
    else:
        signal = signal

    # Reset random seed (so it doesn't affect global)
    np.random.seed(None)
    return signal


def signal_resample(
    signal,
    desired_length=None,
    sampling_rate=None,
    desired_sampling_rate=None
):
    """
    Description:
        Resample a continuous signal to a different length or sampling rate
    Args:
        signal: signal in the form of a vector of values.
        desired_length: desired length of the signal.
        sampling_rate: original sampling frequency
        desired_sampling_rate : desired sampling frequency
    Returns:
        resampled: a vector containing resampled signal values.
    """
    if desired_length is None:
        desired_length = int(np.round(len(signal) * desired_sampling_rate / sampling_rate))

    # Sanity checks
    if len(signal) == desired_length:
        return signal

    # Resample
    resampled = scipy.ndimage.zoom(signal, desired_length / len(signal))
    
    return resampled


def add_respiration(dataset):
    num_points = 10 * 100
    x_space = np.linspace(0,1,num_points)
    for i in range(dataset.shape[0]):
        rr = random.randint(12,20)
        seg_fre = rr / (60 / 10)
        seg_amp = 0.5
        rr_component = seg_amp * np.sin(2 * np.pi * seg_fre * x_space)
        dataset[i,:1000] = dataset[i,:1000] * (rr_component + 1)
        dataset[i,-4] = rr
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

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def white_noise(data, peaks, mag, low_frequency, high_frequency):
    noise = mag * np.random.uniform(-1,1,1000)
    b, a = butter_bandpass(low_frequency, high_frequency, 100, order=5)
    high_freq_noise = filtfilt(b, a, noise)
    # plt.plot(high_freq_noise)
    # plt.show()
    if len(peaks) % 2 != 0:
        raise ValueError("The intervals array length must be even.")
    for i in range(0, len(peaks), 2):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        # print(start_idx)
        # print(end_idx)
        if start_idx < 0 or end_idx >= len(data):
            raise ValueError("Interval indices are out of bounds.")
        data[start_idx:end_idx + 1] += high_freq_noise[start_idx:end_idx + 1]

    return data