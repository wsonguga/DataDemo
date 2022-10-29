#!/usr/bin/env python3
# Source: https://medium.com/@ansjin/dimensionality-reduction-using-pca-on-multivariate-timeseries-data-b5cc07238dc4

# Step1 : Import the required libraries
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# Load NeuroKit and other useful packages
import sys
import pandas as pd # for using pandas daraframe
import numpy as np # for som math operations
from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation
import matplotlib.pyplot as plt # for plotting
import random
sys.path.insert(1, './')
import datasim as nk

plt.rcParams['figure.figsize'] = [20, 8]  # Bigger images

# Step2 : Read the dataset
# df = pd.read_excel('m_1.xlsx', index_col = 0)
# df.drop('machine_id', axis=1, inplace=True)
# df = df.fillna(0)
# df.head()
# df.shape
# X = df.values # getting all values as a matrix of dataframe 

def sim_scg(N, duration, heart_rate, systolic, diastolic, noise):
    N = 20
    duration = 10
    simulated_data = []
    noise = noise

    # plt.figure(figsize=(20, 4))

    for ind in range(N):
        heart_rate = random.randint(heart_rate, heart_rate+10)
        systolic = random.randint(systolic,systolic+10)
        diastolic = random.randint(diastolic,diastolic+10)
        fs = 100

        data = nk.scg_simulate(duration=duration, sampling_rate=fs, noise=noise, heart_rate=heart_rate, systolic=systolic, diastolic=diastolic)
        simulated_data.append(list(data)+[heart_rate]+[systolic]+[diastolic])
    return (simulated_data)

heart_rates = [60, 60, 80, 80, 80]
systolics = [120, 120, 130, 130, 130]
diastolics = [60, 60, 80, 80, 80]

noises = [0.15, 0.15, 0.15, 0.15, 0.15]
N = len(noises)

plt.figure(figsize=(20, 8))

for ind in range(N):
    X = sim_scg(20, 10, heart_rates[ind], systolics[ind], diastolics[ind], noises[ind])
    sc = StandardScaler() # creating a StandardScaler object
    X_std = sc.fit_transform(X) # standardizing the data
    num_components = 1
    pca = PCA(num_components)
    X_pca = pca.fit_transform(X_std) # fit and reduce dimension
    # plt.imshow(X_std)
    # plt.show()
    plt.subplot(N,1,ind+1)
    plt.plot(X_pca)
    plt.title('Noise %f' % (noises[ind]))
    plt.ylabel('Amplitude')
    plt.xlabel('Number of Samples')

plt.show()

# X = sim_scg(20, 10, 0.15)
# # Step3 : Standardizing the data

# sc = StandardScaler() # creating a StandardScaler object
# X_std = sc.fit_transform(X) # standardizing the data

# # # Step4 : Apply PCA¶
# # pca = PCA()
# # X_pca = pca.fit(X_std)

# # # Step5 : Determine the number of components
# # plt.plot(np.cumsum(pca.explained_variance_ratio_))
# # plt.xlabel('number of components')
# # plt.ylabel('cumulative explained variance')
# # plt.show()

# # Step6 : Dimensionality Reduction
# num_components = 1
# pca = PCA(num_components)  
# X_pca = pca.fit_transform(X_std) # fit and reduce dimension
# # plt.imshow(X_std)
# # plt.show()
# plt.plot(X_pca)
# plt.show()

# # Step4-6 together (optional) : Dimensionality Reduction
# pca = PCA(n_components = 0.99)
# X_pca = pca.fit_transform(X_std) # this will fit and reduce dimensions
# print(pca.n_components_) # one can print and see how many components are selected. In this case it is 4 same as above we saw in step 5

# # Step7 : Finding the most important features set
# # pd.DataFrame(pca.components_, columns = df.columns)
# n_pcs= pca.n_components_ # get number of component
# # get the index of the most important feature on EACH component
# most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
# # initial_feature_names = df.columns
# # get the most important feature names
# # most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
# # print(most_important_names)
