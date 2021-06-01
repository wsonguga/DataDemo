#!/usr/bin/env python3

# %%
import warnings
warnings.filterwarnings('ignore')
# Load NeuroKit and other useful packages
import sys
import time
sys.path.insert(1, './')
# import datasim as nk
import numpy as np
# import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import random
# from scipy import signal
# #%matplotlib inline
# from tslearn.clustering import KShape
# from tslearn.datasets import CachedDatasets
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams['figure.figsize'] = [20, 7]  # Bigger images

pause = False

def onclick(event):
    global pause 
    if event.dblclick:
        quit()
    else:
        pause = not pause
        # print(pause)

def on_close(event):
    print('Closed Figure!')
    quit()

def plot_data_set(data ,labels):
    N = data.shape[0]
    # N = 20 # data_set.shape[0]
    R = 10 #10 # num of rows
    C = 2 # num of cols

    fig, ax = plt.subplots()
    # ax.plot(np.random.rand(10))

    k = 0
    total = 0
    while(True): 
        if(total >= N):
            break
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('close_event', on_close)

            # plt.text(0, 0, "Single click to pause/unpause, double click to quit")
        fig.text(0.3, 0.95, "Single click to pause/unpause, double click to quit. Do not click on window close icon", size=10, rotation=0,
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                    )
            )
        for col in range(C):
            if(total >= N):
                    break
            for ind in range(R):
                if(total >= N):
                    break
                index_row = k*C*R + col*R+ ind
                if labels[index_row][-1] == 1:
                    color = 'k-' # bad data
                else:
                    color = 'g-' # good data
                plt.subplot(R,2,2*ind+col+1)
                plt.cla()   # clear previous plot
                plt.plot(data[index_row], color)
                plt.title(f'index: [{index_row}] labels: {labels[index_row].astype(int)}')
                plt.ylabel('Amplitude')
                plt.xlabel('Number of Samples')
                total += 1

        k += 1
        plt.pause(0.05)
        while(pause):
            plt.pause(0.05)

    plt.show()

if(len(sys.argv) > 1):
    file = sys.argv[1] #http
    file_list = [ [file] ]
else:
    file_list = [
            ['./data/classifier_train_data.npy'],
            # ['./data/training_set.npy'], #5000
            # ['./data/good_set.npy'], #1500
            # ['./data/bad_set.npy'], #1500
            # ['./data/mixed_set.npy'],
            ]

for file in file_list:
    data_set = np.load(file[0])
    print(data_set.shape)
    N = data_set.shape[0]
    M = 10

    data = data_set[:,0:1000]
    labels = data_set[:,1000:]

    plot_data_set(data, labels)