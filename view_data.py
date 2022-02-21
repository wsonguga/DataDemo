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
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import random
from matplotlib.backend_bases import MouseButton
# from utils import *

def load_data_file(data_file):
    if data_file.endswith('.csv'):
        data_set = pd.read_csv(data_file).to_numpy()
    elif data_file.endswith('.npy'):
        data_set = np.load(data_file)
    return data_set

plt.rcParams['figure.figsize'] = [20, 8]  # Bigger images

plot_data_index = 0
max_data_index = 0
is_paused = True

def onclick(event):
    global plot_data_index, max_data_index
    global is_paused 

    if event.dblclick:
        is_paused = not is_paused
    elif event.button is MouseButton.LEFT:
        plot_data_index -= 1
        if plot_data_index<0: 
            plot_data_index = 0
    else:
        plot_data_index += 1
        if plot_data_index >= max_data_index:
            plot_data_index = max_data_index

def on_close(event):
    # print('Closed Figure!')
    quit()

def plot_data_set(data,labels):
    global plot_data_index
    global is_paused

    # fs = 100; duration = 10
    # size = fs * duration
    # N = size
    # nfft = np.power( 2, int(np.ceil(np.log2(N))) )
    # NW = 46 #50
    # (dpss, eigs) = nt_alg.dpss_windows(N, NW, 2*NW)
    # keep = eigs > 0.9
    # dpss = dpss[keep]; eigs = eigs[keep]
    # fm = int( np.round(float(200) * nfft / N) )
    # mpdEnv = 20

    N = data.shape[0]

    # N = 20 # data_set.shape[0]
    R = 5 #10 # num of rows
    C = 2 # num of cols
    plot_data_index = int(random.randint(0, N)/(R*C))
    max_data_index = int(N/(R*C))

    fig, ax = plt.subplots()
    # ax.plot(np.random.rand(10))

    k = plot_data_index
    total = 0
    while(True): 
        if(total >= N):
            break
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('close_event', on_close)

            # plt.text(0, 0, "Single click to pause/unpause, double click to quit")
        fig.text(0.3, 0.95, f"[Paused: {is_paused}]; double_click: pause/unpause; left_click: slide_left; right_click: slide_right", size=10, rotation=0,
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
                x = data[index_row]
                # x = dsp.butter_lowpass_filter(x, 5, fs, 2)

                # good_quality, quality_label = dsp.calc_vital_qc(x, fs=100)
                # if good_quality:
                #     color = 'g' # good data
                # else:
                #     color = 'k' # bad data
                ax = plt.subplot(R,2,2*ind+col+1)

                ax.cla()   # clear previous plot
                ax.plot(x, color = 'g')

                signal = x

                ax.set_title(f'index: [{index_row}] labels: {labels[index_row, :].astype(int)}')
                ax.get_xaxis().set_visible(False)
                total += 1

        plt.pause(0.3)
        if(not is_paused):
            plot_data_index += 1
            if plot_data_index >= max_data_index:
                plot_data_index = max_data_index
            k = plot_data_index
        else:
            while (k == plot_data_index):
                plt.pause(0.3)
            k = plot_data_index
        
    plt.show()



if __name__ == '__main__':
    if(len(sys.argv) > 2):
        data_file = sys.argv[1] #htt
        num_labels = int(sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} path_file num_labels")
        print(f"Example: {sys.argv[0]} ../../data/all.1000_6.npy 6")        

        exit()
        
    # vital_indexes = [('H', -4), ('R', -3), ('S', -2), ('D', -1)]
    # data_set = np.load(data_file)
    data_set = load_data_file(data_file)

    data = data_set[:,0:-num_labels]
    labels = data_set[:,-num_labels:]

    print(data_set.shape, data.shape, labels.shape)

    plot_data_set(data, labels)
    exit()