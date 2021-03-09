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

plt.rcParams['figure.figsize'] = [20, 4]  # Bigger images

good_data = np.load("./data/scg/good_data.npy")
bad_data = np.load("./data/scg/bad_data.npy")

print(good_data.shape)

print(bad_data.shape)

from scipy import spatial
from sklearn.metrics.pairwise import pairwise_distances

# anchor = good_data[0]

# for data in good_data:
#     distance = spatial.distance.cosine(data, anchor)
#     print("good:", distance)
#     # distance = cosine_distances(data, anchor)
#     # print("good:", distance)

# for data in bad_data:
#     distance = spatial.distance.cosine(data, anchor)
#     print("bad:", distance)
#     # distance = cosine_distances(data, anchor)
#     # print("bad:", distance)


all_data = arr = np.concatenate((good_data, bad_data), axis=0)

print(all_data.shape)

N = 20
for ind in range(N):
    # plt.figure(figsize=(20, 4))
    plt.subplot(N,2,2*ind+1)
    plt.plot(good_data[ind])
    plt.title('Good')
    plt.ylabel('Amplitude')
    plt.xlabel('Number of Samples')

    plt.subplot(N,2,2*ind+2)
    plt.plot(bad_data[ind])
    plt.title('Bad')
    plt.ylabel('Amplitude')
    plt.xlabel('Number of Samples')

plt.show()


# distance = pairwise_distances(all_data, all_data, metric='manhattan')
# print("manhattan:", distance)

# plt.imshow(distance)
# plt.show()

distance = pairwise_distances(all_data, all_data, metric='cosine')
# print("cosine:", distance)
plt.imshow(distance)
plt.show()


distance = np.corrcoef(all_data)
plt.imshow(distance)
plt.show()

