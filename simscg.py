#!/usr/bin/env python3

# %%
import warnings
warnings.filterwarnings('ignore')
# Load NeuroKit and other useful packages
import sys
sys.path.insert(1, './')
import datasim as nk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from scipy import signal
#%matplotlib inline

plt.rcParams['figure.figsize'] = [20, 10]  # Bigger images

# gen_list = [
#             ['training_set',5000, 0.15], #5000
#             ['good_set',500, 0.15], #1500
#             ['bad_set',500, 0.75], #1500
#             ['mixed_set',500, -1],
#             ]

gen_list = [
            ['training_set',5000, 0], #5000
            ]

for gen_info in gen_list:
    # import pdb; pdb.set_trace()
    file_name = gen_info[0]
    N = int(gen_info[1])

    duration = 10
    simulated_data = []

    # plt.figure(figsize=(20, 4))

    for ind in range(N):
        heart_rate = random.randint(60, 90)
        respiratory_rate = random.randint(10, 25)
        systolic = random.randint(90,150)
        diastolic = random.randint(70,100)
        fs = 100

        noise = float(gen_info[2])
        if noise == -1:
            noise = random.choice([0.15, 0.5, 0.75])
            # print (noise)

        data = nk.scg_simulate(duration=duration, sampling_rate=fs, noise=noise, heart_rate=heart_rate, respiratory_rate=respiratory_rate, systolic=systolic, diastolic=diastolic)
        simulated_data.append(list(data)+[heart_rate]+[systolic]+[diastolic])

        # plot the signals and spectrogram
        x = data #/np.linalg.norm(data)
        fmin = 0
        fmax = 10
        npseg = 200
        slidingstep=20

        # import pdb; pdb.set_trace()


        # plt.subplot(N,2,2*ind+1)

        # plt.subplot(N,1,ind+1)

        # plt.plot(x)
        # plt.title(f'Raw, HR {heart_rate} SBP {systolic} DBP {diastolic} Noise {noise}')
        # plt.ylabel('Amplitude')
        # plt.xlabel('Number of Samples')

        # plot spectrogram
        # f, t, Zxx = signal.stft(x, fs, nperseg=npseg, noverlap = npseg-slidingstep)
        # f, t, Zxx = signal.spectrogram(x, fs)

        # plt.subplot(N,2,2*ind+2)
        # plt.pcolormesh(t, f[int(fmin*npseg/fs):int(fmax*npseg/fs)+1], np.abs(Zxx[int(fmin*npseg/fs):int(fmax*npseg/fs)+1,:]), shading='gouraud')
        # #plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        # plt.title(f'STFT, HR {heart_rate} SBP {systolic} DBP {diastolic}')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        ind +=1

    # plt.show()
    simulated_data = np.asarray(simulated_data)
    np.save(f'./data/{file_name}',simulated_data)
    print(f'{file_name} is generated and saved!')
    # import pdb; pdb.set_trace()
