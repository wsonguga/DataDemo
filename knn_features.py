#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
sns.set()

import tsfel
from sklearn.neighbors import KNeighborsRegressor
from joblib import dump, load
import random
import os
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
from random import sample
from math import isnan
import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


plt.rcParams['figure.figsize'] = [20, 7]  # Bigger images

pause = False

def onclick(event):
    global pause 
    if event.dblclick:
        quit()
    else:
        pause = not pause
        # print(pause)

def generate_sinwave():
    fs = 100 # sampling frequency
    fd = 1 # signal fundamental frequencys
    t = 10
    samples = np.linspace(0, t, t * fs)
    wave = np.sin(2 * np.pi * fd * samples)
    return wave


### example
# some_data = np.load('path/to/data.npy')
# result_ = extract_specific_bp(some_data, num_extraction=2, num_row=-2) ### -2 is SBP,  -1 is DBP in that data order.

def extract_specific_bp(data, num_extraction, index_col):
    '''
    imports:
        import numpy as np
        import random
    args:
        data --- array like data
        num_extraction --- how many to extract
        index_col --- the index of target label located in array
    return:
        result --- array like data including targets
    '''
    range_set = set(data[:,index_col])    ## get all target values
    print(len(range_set), range_set)

    result = []
    i = 0
    for each_target in range_set:
        ### if there is not enough data for requesting, skip
        if len(np.where(data[:,index_col]==each_target)[0]) < num_extraction:
            print(f'For target {each_target}, there is only {len(np.where(data[:,index_col]==each_target)[0])} data, however, you request {num_extraction}. Skip!')
            continue

        ### randomly get indexs of request target
        index = random.sample(list(np.where(data[:,index_col]==each_target)[0]), num_extraction)
        ### append requested target to result
        # result.append(data[index,:])
        if i ==0:
            result = data[index,:] 
        else:
            result = np.append(result, data[index,:] , axis=0)
        i += 1


    return np.array(result)

def feature_reduction(signals, feature_reduction_model):
    '''
    input:
        signal has size (n,1000)
    output:
        feautres has size (n,97)
    '''

    cfg = tsfel.get_features_by_domain()


    window_mean = []
    window_median = []
    window_max = []
    window_min = []
    window_std = []
    for i in range(signals.shape[0]):
        print(i)
        features = tsfel.time_series_features_extractor(cfg, signals[i], window_size = 100)
        mean = features.mean()
        med = features.median()
        mini = features.min()
        maxi = features.max()
        sd = features.std()
        window_mean.append(mean)
        window_median.append(med)
        window_max.append(maxi)
        window_min.append(mini)
        window_std.append(sd)


    window_all_feat = pd.concat([pd.DataFrame(window_median), pd.DataFrame(window_mean), pd.DataFrame(window_min), pd.DataFrame(window_max), pd.DataFrame(window_std) ], axis= 1)

    scaler = preprocessing.StandardScaler()
    scaler.fit(window_all_feat)
    window_all_feat = scaler.transform(window_all_feat)

    X_reduced = feature_reduction_model.transform(window_all_feat)

    return X_reduced

def extract_windowed_features(data, scaler):
    cfg = tsfel.get_features_by_domain()

    window_mean_vali1 = []
    window_median_vali1 = []
    window_max_vali1 = []
    window_min_vali1 = []
    window_std_vali1 = []


    features_vali1 = tsfel.time_series_features_extractor(cfg, data, window_size = 100, verbose=False)
    mean_vali1 = features_vali1.mean()
    med_vali1 = features_vali1.median()
    mini_vali1 = features_vali1.min()
    maxi_vali1 = features_vali1.max()
    sd_vali1 = features_vali1.std()
    window_mean_vali1.append(mean_vali1)
    window_median_vali1.append(med_vali1)
    window_max_vali1.append(maxi_vali1)
    window_min_vali1.append(mini_vali1)
    window_std_vali1.append(sd_vali1)

    extracted_features = pd.concat([pd.DataFrame(window_median_vali1), pd.DataFrame(window_mean_vali1), pd.DataFrame(window_min_vali1), pd.DataFrame(window_max_vali1), pd.DataFrame(window_std_vali1) ], axis= 1)
    # extracted_features.drop(self.corr_features, axis=1, inplace=True)
    # extracted_features = self.selector.fit_transform(extracted_features)
    extracted_features = scaler.transform(extracted_features)
    print(extracted_features)
    return extracted_features

def extract_features(data, scaler):
    cfg = tsfel.get_features_by_domain()

    extracted_features = tsfel.time_series_features_extractor(cfg, data, verbose=False)
   
    # extracted_features = scaler.transform(extracted_features)
    print(*extracted_features, sep = ", ")
    return extracted_features

# def feature_extraction(data_fe):
#   cfg = tsfel.get_features_by_domain()
#   features = tsfel.time_series_features_extractor(cfg, data_fe)
#   features.replace([np.inf, -np.inf], np.nan, inplace=True)
#   features.fillna(features.median(), inplace=True)
#   return features

if(len(sys.argv) > 1):
    file = sys.argv[1] #http
    file_list = [ [file] ]
else:
    file_list = [
            # ['./data/classifier_train_data.npy'],
            ['./data/b8_27_eb_80_1c_cf_80_200_mac.npy'],
            # ['./data/training_set.npy'], #5000
            # ['./data/good_set.npy'], #1500
            # ['./data/bad_set.npy'], #1500
            # ['./data/mixed_set.npy'],
            ]


save_path = "./model/"
# modelKNN, scaler = load(os.path.join(save_path,"knn_model_w_scaler.joblib"))

model = load(os.path.join(save_path,'feature_to_97.joblib'))

for file in file_list:
    data_set = np.load(file[0])
    print(data_set.shape)
    N = 20 # data_set.shape[0]
    M = 10

    # data = data_set[0:N,0:1000]
    # labels = data_set[0:N,1000:]

    data_set = extract_specific_bp(data_set, num_extraction=4, index_col=-2) 
    N = data_set.shape[0]
    data = data_set[0:N,0:1000]
    labels = data_set[0:N,1000:]

    print(data.shape)

    # exit()

    features = feature_reduction(data, model)
    np.save("./data/scg_feature.npy", features)
    # features = []
    # for i in range(N):
    #     # print(data[i])
    #     signals = np.reshape(np.asarray(data[i]),(1, 1000))
    #     # print(signals.shape[0])
    #     feature = feature_reduction(signals, model)
    #     print(*feature)
    #     # feature = extract_features(data[i], scaler)
    #     features.append(feature) 

    fig, ax = plt.subplots()
    # ax.plot(np.random.rand(10))

    for k in range(int(N/(2*M))): 
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
            # plt.text(0, 0, "Single click to pause/unpause, double click to quit")
        fig.text(0.3, 0.95, "Single click to pause/unpause, double click to quit. Do not click on window close icon", size=10, rotation=0,
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                    )
            )
        for ind in range(M):
            # plt.figure(figsize=(20, 4))
            # fig, ax = plt.subplot(M,2,2*ind+1)
            # ax.cla()   # clear previous plot
            # ax.plot(data[k*2*M + 2*ind])
            # ax.title(f'index: [{k*2*M + 2*ind}] Parameters: {parameters[k*2*M + 2*ind]}')
            # ax.ylabel('Amplitude')
            # ax.xlabel('Number of Samples')
            # cid = fig.canvas.mpl_connect('button_press_event', onclick)

            index = k*2*M + 2*ind
            if labels[index][-1] == 1:
                color = 'k-' # bad data
            else:
                color = 'g-' # good data
            plt.subplot(M,2,2*ind+1)
            plt.cla()   # clear previous plot
            plt.plot(features[index], color)
            plt.title(f'index: [{index}] labels: {labels[index].astype(int)}')
            plt.ylabel('Amplitude')
            plt.xlabel('Number of Samples')

            index = k*2*M + 2*ind + 1
            if labels[index][-1] == 1: 
                color = 'k-' # bad data
            else:
                color = 'g-' # good data            
            plt.subplot(M,2,2*ind+2)
            plt.cla()   # clear previous plot
            plt.plot(features[index], color)
            plt.title(f'index: [{index}] labels: {labels[index].astype(int)}')
            plt.ylabel('Amplitude')
            plt.xlabel('Number of Samples')
        plt.pause(0.05)
        while(pause):
            plt.pause(0.05)

    plt.show()