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
from tslearn.clustering import KShape
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams['figure.figsize'] = [20, 8]  # Bigger images

good_data = np.load("./data/scg/good_data.npy")
bad_data = np.load("./data/scg/bad_data.npy")

print(good_data.shape)

print(bad_data.shape)

from scipy import spatial
from sklearn.metrics.pairwise import pairwise_distances
from data_quality_classifier import DATA_QUALITY_Model

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

all_data = arr = np.load('./data/scg/F1_0204_all_data.npy')
np.random.shuffle(all_data)
all_data = arr = all_data[:60,:1000]
# import pdb; pdb.set_trace()
#
# N = 20
# for ind in range(N):
#     # plt.figure(figsize=(20, 4))
#     plt.subplot(N,2,2*ind+1)
#     plt.plot(good_data[ind])
#     plt.title('Good')
#     plt.ylabel('Amplitude')
#     plt.xlabel('Number of Samples')
#
#     plt.subplot(N,2,2*ind+2)
#     plt.plot(bad_data[ind])
#     plt.title('Bad')
#     plt.ylabel('Amplitude')
#     plt.xlabel('Number of Samples')
#
# plt.show()

distance = np.corrcoef(all_data)
plt.imshow(distance)
plt.show()
print(distance)

# distance = pairwise_distances(all_data, all_data, metric='manhattan')
# print("manhattan:", distance)

# plt.imshow(distance)
# plt.show()

# distance = pairwise_distances(all_data, all_data, metric='cosine')
# # print("cosine:", distance)
# plt.imshow(distance)
# plt.show()

# In[2]
np.random.shuffle(all_data)
print(all_data.shape)

# For this method to operate properly, prior scaling is required
x_train = TimeSeriesScalerMeanVariance().fit_transform(all_data)
sz = x_train.shape[1]

# kShape clustering
seed = 0
ks = KShape(n_clusters=2, verbose=True, random_state=seed)
y_pred = ks.fit_predict(x_train)

print(x_train.shape)
print(y_pred.shape)

plt.figure()
for yi in range(2):
    N = len(x_train[y_pred == yi])
    ind = 0
    for xx in x_train[y_pred == yi]:
        plt.subplot(N, 2, 2*ind+yi+1)
        plt.plot(xx)
        ind += 1
        # plt.plot(xx.ravel(), "k-", alpha=.2)
    # plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    # plt.xlim(0, sz)
    # plt.ylim(-4, 4)
    plt.title("Cluster %d, N %d" % (yi + 1, N))
    plt.tight_layout()



#### classifier
data_quality_model = DATA_QUALITY_Model()
# data_quality_model.fit(all_data=dataset, window_len=1000, devide_factor=0.8,learning_rate=0.0005, batch_size=64, dim_feature=500)
data_quality_model.load_model('./Data_Quality_Classifier_models')

good_data = []
bad_data = []
for jnd, each_vali in enumerate(all_data):
    cur_pred = data_quality_model.predict(each_vali)
    # if (len(good_data) >= 40) and  (len(bad_data) >= 40):
    #     break
    if cur_pred == 0:
        good_data.append(each_vali[:1000])
        # plt.plot(each_vali[:1000])
        # plt.show()
        # pdf_good.savefig()
        # plt.cla()
    else:
        bad_data.append(each_vali[:1000])
        # plt.plot(each_vali[:1000])
        # plt.show()
        # pass
        # pdf_bad.savefig()
        # plt.cla()
good_data = np.asarray(good_data)
bad_data = np.asarray(bad_data)
# import pdb; pdb.set_trace()

plt.figure()
for gnd, each_good in enumerate(good_data):
    plt.subplot(len(good_data),1,gnd+1)
    plt.plot(each_good)
    # plt.
plt.figure()
for bnd, each_bad in enumerate(bad_data):
    plt.subplot(len(bad_data),1,bnd+1)
    plt.plot(each_bad)


plt.show()
# %%
