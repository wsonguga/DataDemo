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

    if(len(sys.argv) > 2):
        data_file = sys.argv[1] #htt
        N = int(sys.argv[2])
        noise = float(sys.argv[3])
    else:
        print(f"Usage: {sys.argv[0]} path_file num_rows noise_level \n where noise level (amplitude of the laplace noise).")
        print(f"Example: {sys.argv[0]} ../data/simu.1000_6.npy 100 0.5")       
        exit()

    fs = 100
    duration = 10 # 10 seconds
    simulated_data = []

    for ind in tqdm(range(N)):
        heart_rate = random.randint(50, 150)
        respiratory_rate = random.randint(10, 30)

        systolic = random.randint(90, 180)
        diastolic = systolic - random.randint(40,45) #+ systolic

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
