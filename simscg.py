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
from tqdm import tqdm

if __name__ == '__main__':

    if(len(sys.argv) >= 5):
        N = int(sys.argv[1])
        noise = float(sys.argv[2])
        S_min = int(sys.argv[3])
        S_max = int(sys.argv[4])
        data_file = sys.argv[5] #htt

    else:
        print(f"Usage: {sys.argv[0]} num_rows noise_level S_min S_max path_file \n where noise level (amplitude of the laplace noise).")
        print(f"Example: {sys.argv[0]} 100 0.5 90 180 ../data/simu.1000_6.npy")       
        exit()

    fs = 100
    duration = 10 # 10 seconds
    simulated_data = []

    for ind in tqdm(range(N)):
        heart_rate = random.randint(50, 150)
        respiratory_rate = random.randint(10, 30)

        systolic = random.randint(S_min, S_max)
        diastolic = random.randint(60,100) #+ systolic

        # systolic = random.randint(90, 170)
        # diastolic = random.randint(50,110)
        # while (systolic - diastolic > 60 or systolic - diastolic < 20):
        #     diastolic = random.randint(50,110)

        print('hr:', heart_rate, 'rr:', respiratory_rate, 
              'sp:', systolic, 'dp:', diastolic)
       
        data = nk.scg_simulate(duration=duration, sampling_rate=fs, noise=noise, heart_rate=heart_rate, respiratory_rate=respiratory_rate, systolic=systolic, diastolic=diastolic)
        ## N + 6 size. 6 are [mat_int(here 0 for synthetic data), time_stamp, hr, rr, sbp, dbp]
        simulated_data.append(list(data)+[0]+[ind]+[heart_rate]+[respiratory_rate]+[systolic]+[diastolic])

    # plt.show()
    simulated_data = np.asarray(simulated_data)
    np.save(data_file,simulated_data)
    print(f'{data_file} is generated and saved!')
    # import pdb; pdb.set_trace()
