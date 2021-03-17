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

gen_list = [
            ['training_set',5000],
            ['test_set',1500],
            ]

for gen_info in gen_list:
    # import pdb; pdb.set_trace()
    file_name = gen_info[0]
    N = int(gen_info[1])
    duration = 10
    simulated_data = []

    for ind in range(N):
        heart_rate = random.randint(60, 90)
        systolic = random.randint(90,150)
        diastolic = random.randint(70,100)
        fs = 100

        data = nk.scg_simulate(duration=duration, sampling_rate=fs, noise=0.15, heart_rate=heart_rate, systolic=systolic, diastolic=diastolic)
        x = data #/np.linalg.norm(data)

        fmin = 0
        fmax = 10
        npseg = 200
        slidingstep=20

        simulated_data.append(list(data)+[heart_rate]+[systolic]+[diastolic])

        # import pdb; pdb.set_trace()

        # plt.figure(figsize=(20, 4))
        # plt.subplot(N,2,2*ind+1)

        # plt.subplot(N,1,ind+1)
        #
        # plt.plot(x)
        # plt.title(f'Raw, HR {heart_rate} SBP {systolic} DBP {diastolic}')
        # plt.ylabel('Amplitude')
        # plt.xlabel('Number of Samples')



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

    # # Alternate heart rate and noise levels
    # scg100 = nk.scg_simulate(duration=10, noise=0.05, heart_rate=60, systolic=100, method="simple")
    # scg120 = nk.scg_simulate(duration=10, noise=0.05, heart_rate=60, systolic=120, method="simple")
    # scg140 = nk.scg_simulate(duration=10, noise=0.05, heart_rate=60, systolic=140, method="simple")

    # # scg100 = nk.scg_simulate(duration=10, noise=0.05, heart_rate=60, method="scgsyn")

    # # Visualize
    # # scg_df = pd.DataFrame({"SCG_100": scg100,
    # #                        "SCG_50": scg50})
    # scg_df = pd.DataFrame({"SCG_100": scg100,
    #                        "SCG_120": scg120,
    #                        "SCG_140": scg140})


    # nk.signal_plot(scg_df, subplots=True)
    # plt.show()

    # In[3]

    # %%
