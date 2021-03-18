#!/usr/bin/env python3
'''
author: ming
ming.song.cn@outlook.com
copyright@2020
[TODO] model save as one file, remember to add window_len to model file
        warm start method
'''


import os
import sys
import numpy as np
import torch
from torch.optim import *
from torch import nn, optim, cuda
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data import
from sklearn import preprocessing
import random
import copy
from random import sample
from math import isnan
import datetime
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
# import scgkit2 as ds
# batch_size = 1000
# test_only = False
# VISUAL_FLAG = False
# # test_only = bool(sys.argv[1])
# lr = 0.001
# dim_feature = 100

# number of labels
# NL = 3

class Initial_Dataset(Dataset):
    """docstring for ."""

    def __init__(self, data, data_length,dim_feature=100, scalers=None):  # before this length is data, after is label
        self.data = data
        self.data_length = data_length
        self.dim_feature = dim_feature

        if scalers == None:

            self.scaler_x = preprocessing.StandardScaler().fit(self.data[:,:self.data_length])
            self.scaler_x.scale_[:] = np.std(self.data[:,:self.data_length])
            self.scaler_x.mean_[:] = np.mean(self.data[:,:self.data_length])

            self.array_Tx_temp = self.scaler_x.transform(self.data[:,:self.data_length])
            self.array_Tx_temp = self.array_Tx_temp[:, :(len(self.array_Tx_temp[0]) // self.dim_feature) * self.dim_feature]
            self.array_Tx = self.array_Tx_temp.reshape(len(self.array_Tx_temp),-1,self.dim_feature)


            self.scaler_y = preprocessing.StandardScaler().fit(self.data[:,-NL:])
            self.array_Ty = self.scaler_y.transform(self.data[:,-NL:])
            # self.array_Ty = self.data[:,-4:]

        else:
            self.scaler_x = scalers[0]
            self.scaler_y = scalers[1]
            self.array_Tx_temp = self.scaler_x.transform(self.data[:,:self.data_length])
            self.array_Tx_temp = self.array_Tx_temp[:, :(len(self.array_Tx_temp[0]) // self.dim_feature) * self.dim_feature]
            self.array_Tx = self.array_Tx_temp.reshape(len(self.array_Tx_temp),-1,self.dim_feature)

            self.array_Ty = self.scaler_y.transform(self.data[:,-NL:])
            # self.array_Ty = self.data[:,-4:]


    def __getitem__(self, index):
        data_ = self.array_Tx[index, :]
        gt_ = self.array_Ty[index, :] #

        return data_, gt_

    def get_scalers(self):
        return self.scaler_x, self.scaler_y

    def __len__(self):
        return self.data.shape[0]


class LstmAttentionNet(nn.Module):
    def __init__(self):
        super(LstmAttentionNet, self).__init__()
        hidden_size = 60
        attention_size = hidden_size
        self.lstm = nn.LSTM(input_size=100, hidden_size=hidden_size, batch_first=True, num_layers=3)
        self.w_omega = nn.Parameter(torch.randn(hidden_size,attention_size))
        self.b_omega = nn.Parameter(torch.randn(attention_size))
        self.u_omega = nn.Parameter(torch.randn(attention_size,1))
        self.decoding_layer = nn.Linear(hidden_size, NL)

        self.out = nn.Linear(attention_size, 60)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        v = torch.matmul(out,self.w_omega)+self.b_omega
        vu = torch.matmul(v, self.u_omega)
        weight= nn.functional.softmax(vu,dim=1)
        out_weighted = torch.sum(out*weight,1)
        y_pred = self.decoding_layer(out_weighted)
        # import pdb; pdb.set_trace()
        # y_pred = self.out(out_weighted)
        # y_pred = self.softmax(pred)

        return y_pred#, weight


class DL_Model():
    """docstring for DL_Model."""

    def __init__(self, LOG=False):
        super(DL_Model, self).__init__()
        ####
        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')

        # self.ae_Net = AE()
        # self.cnn_Net = CNN_classification()
        self.lstmAtt_Net = LstmAttentionNet()
        # self.ae_Net = Batch_Net_2(3000,256,128,64,32)
        # self.ae_Net = Batch_Net(3000,256,128)
        # self.ae_Net = self.ae_Net.to(device = self.device)
        # self.cnn_Net = self.cnn_Net.to(device = self.device)
        self.lstmAtt_Net = self.lstmAtt_Net.to(device = self.device)
        # self.ae_Net = nn.DataParallel(self.ae_Net,device_ids=[0,1,2,3]) # multi-GPU

        print(f"Using device:{self.device}")


    def organize_data_from_npy(self, data):
        if data.shape[1] == 2:
            print('Organizing data.')
            dara_re = []
            for each_data in data:
                dara_re.append(np.concatenate((each_data[0][:,1],each_data[1])))
                # import pdb; pdb.set_trace()
            print('Done!')
            return np.asarray(dara_re).astype(np.float)

        else:
            return data.astype(np.float)



    def fit(self, all_data, window_len, devide_factor, dim_feature=100, learning_rate=0.001, batch_size=32, epoch_number=500, CONTINUE_TRAINING = False):
        self.data = all_data
        self.window_len = window_len
        class_num = 60
        # organize data to trainable form
        organized_data = self.organize_data_from_npy(self.data)
        # organized_data = self.data
        # devide train and test data set
        devide_index = int(organized_data.shape[0]*devide_factor)

        # get train and test dataset
        # data_train = organized_data[:devide_index,:]
        # data_test = organized_data[devide_index:,:]

        data_test = organized_data[:devide_index,:]
        data_train = organized_data[devide_index:,:]

        train_dataset = Initial_Dataset(data_train, window_len, dim_feature=dim_feature, scalers=None)
        self.scaler_x, self.scaler_y = train_dataset.get_scalers()
        test_dataset = Initial_Dataset(data_test, window_len, dim_feature=dim_feature, scalers=[self.scaler_x, self.scaler_y])

        # import pdb; pdb.set_trace()

        train_loader = DataLoader(train_dataset, batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        ### training component

        loss_fn_lstmAtt = torch.nn.MSELoss()
        optimizer_lstmAtt = optim.Adam(self.lstmAtt_Net.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer_lstmAtt,step_size=15, gamma = 0.95)

        self.last_error = 1e5
        for e in range(epoch_number):
            # print(e)
            lr = optimizer_lstmAtt.state_dict()['param_groups'][0]['lr']
            print(f'Learning Rate: {lr}')

            for ind, (train_tensor_x, train_tensor_y) in enumerate(train_loader):
                # train_tensor_x = train_tensor_x.squeeze()
                # train_tensor_y = train_tensor_y.squeeze()
                # import pdb; pdb.set_trace()
                # optimizer.zero_grad()
                # if cuda.is_available():
                # train_tensor_x = train_tensor_x.to(self.device)
                # train_tensor_y = train_tensor_y.to(self.device)
                train_tensor_x = torch.tensor(train_tensor_x,dtype=torch.float32,device=self.device)
                train_tensor_y = torch.tensor(train_tensor_y,dtype=torch.float32,device=self.device)

                # train_y_pred_AE, conv2_logit = self.ae_Net(train_tensor_x)
                # train_y_pred_cnn = self.cnn_Net(conv2_logit)
                # import pdb; pdb.set_trace()
                train_y_pred_lstmAtt = self.lstmAtt_Net(train_tensor_x)


                # label = torch.LongTensor(train_tensor_y[:,2:3].long())
                # train_y_label_onehot_cnn = torch.zeros(batch_size,class_num).scatter_(1,label,1)
                # import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
                train_loss_tensor_lstmAtt = loss_fn_lstmAtt(train_y_pred_lstmAtt, train_tensor_y)
                train_loss = train_loss_tensor_lstmAtt.item()

                # if ind == 0:
                #     print(f'Initial train MSE: {train_loss} ')

                # train_loss_tensor_AE = loss_fn_AE(train_y_pred_AE, train_tensor_x)
                # train_loss_tensor_CNN = loss_fn_CNN(train_y_pred_cnn, train_tensor_y[:,2].long())
                # train_loss_tensor = train_loss_tensor_AE + train_loss_tensor_CNN
                # train_loss = train_loss_tensor.item()
                # import pdb; pdb.set_trace()
                # print(train_loss)

                optimizer_lstmAtt.zero_grad()
                train_loss_tensor_lstmAtt.backward()
                optimizer_lstmAtt.step()

                # optimizer_AE.zero_grad()
                # optimizer_CNN.zero_grad()
                # train_loss_tensor.backward()
                # optimizer_AE.step()
                # optimizer_CNN.step()

                # import pdb; pdb.set_trace()
            print(f'Epoch {e} train MSE: {train_loss} ')




            if e % 3 == 0:
                loss_test = []
                for test_tensor_x, test_tensor_y in test_loader:
                    test_tensor_x = torch.tensor(test_tensor_x,dtype=torch.float32,device=self.device)
                    test_tensor_y = torch.tensor(test_tensor_y,dtype=torch.float32,device=self.device)

                    # if cuda.is_available():
                    # test_tensor_x = test_tensor_x
                    # test_tensor_y = test_tensor_y.to(self.device)
                        # test_tensor_x, test_tensor_y = test_tensor_x.cuda(), test_tensor_y.cuda()
                    test_y_pred = self.lstmAtt_Net(test_tensor_x)

                    # test_loss_tensor_lstmAtt = loss_fn_lstmAtt(test_y_pred, test_tensor_y)
                    # test_loss = test_loss_tensor_lstmAtt.item()

                    # test_loss_tensor = loss_fn(test_tensor_y,test_y_pred)
                    # test_loss = test_loss_tensor.item()

                    array_y_pred = self.scaler_y.inverse_transform(test_y_pred.cpu().detach().numpy())
                    array_y_gt = self.scaler_y.inverse_transform(test_tensor_y.cpu().detach().numpy())

                    test_loss = np.mean(np.abs(np.array(array_y_pred)-np.array(array_y_gt)))
                    loss_test.append(test_loss)

                print(f'Epoch {e} test MSE: {np.mean(loss_test)} ')
                self.error = np.mean(loss_test)
                if self.error < self.last_error:
                    if not os.path.exists('./model/LSTM_regression_models'):
                        os.makedirs('./model/LSTM_regression_models')
                        print(f'make dir ./model/LSTM_regression_models')
                    self.save_model(model_path='./model/LSTM_regression_models')
                    self.last_error = self.error


            # learning rate decay
            scheduler.step()
            print('--------------------------------------------------------------')
                # import pdb; pdb.set_trace()


        # import pdb; pdb.set_trace()

    def save_model(self, model_path='./model/LSTM_regression_models'):

        print('save model...')
        with open(os.path.join(model_path,"scaler_param.pk"),"wb+") as f:
            pickle.dump([self.scaler_x,self.scaler_y,self.window_len],f)

        # torch.save(self.ae_Net.state_dict(), os.path.join(model_path,"model_param.pk"),_use_new_zipfile_serialization=False)
        # torch.save(self.ae_Net.state_dict(), os.path.join(model_path,"ae_model_param.pk"))
        # torch.save(self.cnn_Net.state_dict(), os.path.join(model_path,"cnn_model_param.pk"))

        torch.save(self.lstmAtt_Net.state_dict(), os.path.join(model_path,"lstmAtt_model_param.pk"))


        # with open(os.path.join(model_path,"error.pk"),"wb+") as f:
        #     pickle.dump(self.error,f)
        print('save done!')
        # test_error_0 = self.error


    def load_model(self, model_path='./model/LSTM_regression_models'):
        if os.path.exists(os.path.join(model_path,"scaler_param.pk")):
            with open(os.path.join(model_path,"scaler_param.pk"),"rb+") as f:
                [self.scaler_x,self.scaler_y,self.window_len] = pickle.load(f)
        else:
            print(f'scaler_param.pk not exist!')
            quit()

        if os.path.exists(os.path.join(model_path,"lstmAtt_model_param.pk")):
            self.lstmAtt_Net.load_state_dict(torch.load(os.path.join(model_path,"lstmAtt_model_param.pk"),map_location=torch.device(self.device)))
        else:
            print(f'model_param.pk not exist!')
            quit()

        print('Model parameters loaded!')

        # if os.path.exists(os.path.join(model_path,"error.pk")):
        #     with open(os.path.join(model_path,"error.pk"),"rb+") as f:
        #         self.error = pickle.load(f)
        # else:
        #     print(f'error.pk not exist!')
        #     quit()

    # dim_feature=100 means we need to reshape eg 1000 to 10x100, but if the input can’t divided by 100 we just take the part up to n times 100.
    def predict(self, pred_x, dim_feature=100):
        self.pred_x = self.scaler_x.transform([pred_x]) #self.scaler_x.transform([pred_x[:3000]])
        self.pred_x = self.pred_x[:, :(self.window_len // dim_feature) * dim_feature].reshape(len(self.pred_x),-1,dim_feature)

        self.tensor_pred_x = torch.tensor(self.pred_x,dtype=torch.float32,device=self.device)

        self.train_y_pred = self.scaler_y.inverse_transform(self.lstmAtt_Net(self.tensor_pred_x.float()).cpu().detach().numpy())[0]

        # print(self.train_y_pred)

        return self.train_y_pred


    def evaluate(self, data, devide_factor=0.8, VERBOSE=False, INFLUX=False):
        self.data = data
        self.lstmAtt_Net.eval()
        organized_data = self.organize_data_from_npy(self.data)
        devide_index = int(organized_data.shape[0]*devide_factor)

        # get train and test dataset
        data_train = organized_data[:devide_index,:]
        data_test = organized_data[devide_index:,:]

        dataset = Initial_Dataset(data_test, self.window_len, dim_feature=100, scalers=[self.scaler_x, self.scaler_y])

        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        pred_ = []
        gt_ = []

        for tensor_x, tensor_y in test_loader:
            # import pdb; pdb.set_trace()

            # tensor_x = torch.reshape(tensor_x, (-1,self.window_len))
            # import pdb; pdb.set_trace()
            tensor_x = tensor_x.to(self.device)
            tensor_y = tensor_y.to(self.device)

            # y_pred = self.lstmAtt_Net(tensor_x.float()).cpu().detach().numpy().squeeze()
            # input = tensor_x.cpu().detach().numpy().squeeze()

            y_pred = self.scaler_y.inverse_transform(self.lstmAtt_Net(tensor_x.float()).cpu().detach().numpy())[0]
            y_gt = self.scaler_y.inverse_transform(tensor_y.cpu().detach().numpy())[0]
            # y_gt = tensor_y.cpu().detach().numpy()[0]
            # import pdb; pdb.set_trace()

            # plt.figure()
            # plt.subplot(2,1,1)
            # plt.plot(input,'b')
            # plt.subplot(2,1,2)
            # plt.plot(y_pred,'r')
            # plt.show()

            # print(y_pred)

            pred_.append(y_pred)
            gt_.append(y_gt)

        plt.figure(figsize=(20,6))
        plt.plot(pred_,'b')
        plt.plot(gt_,'r')
        plt.show()
        plt.savefig('../result.png')

        pred_ = np.asarray(pred_)
        gt_ = np.asarray(gt_)
        MAE = np.mean(abs(pred_-gt_),axis=0)
        print(f' The MAE is {MAE}')
        # print(f'The MAE of HR: {MAE[0]}, RR: {MAE[1]},SBP: {MAE[2]}, DBP: {MAE[3]}')



def main():
    list_x = []
    list_y = []
    global NL

    ####### modify mapping relation here

    fs = 100
    duration = 10
    # N = 1000
    # for ind in range(N):
    #     heart_rate = random.randint(60, 90)
    #     systolic = random.randint(130,150)
    #     diastolic = random.randint(70,100)
    #     data = ds.scg_simulate(duration=duration, sampling_rate=fs, noise=0.15, heart_rate=heart_rate, systolic=systolic, diastolic=diastolic)
    #     list_x.append(data)
    #     label =  [systolic, diastolic] #, heart_rate]
    #     list_y.append(label)
    data = np.load("./data/training_set.npy")
    # data = np.load("./data/good_set.npy")
    # data = np.load("./data/bad_set.npy")
    # data = np.load("./data/mixed_set.npy")

    NL = 3
    list_x = data[:,:-NL] # 10 second 100Hz SCG data
    list_y = data[:, -NL:] # NL = 3 labels
    # print(list_y)
    # print(list_x, list_y)
    list_x = np.asarray(list_x)
    list_y = np.asarray(list_y)
    dataset = np.concatenate((list_x, list_y),1)


    auto_encoder = DL_Model()
    auto_encoder.fit(all_data=dataset, window_len=duration*fs, devide_factor=0.8,learning_rate=0.0008, batch_size=64, dim_feature=100)
    # auto_encoder.load_model('./model/LSTM_regression_models')
    # auto_encoder.evaluate(dataset, devide_factor=0.8)
    # auto_encoder.evaluate(dataset, devide_factor=0.0)

    # # print(list_x[0,:])
    # pred_y = [0]*list_x.shape[0]
    # # print(list_x.shape[0])
    # for i in range(list_x.shape[0]):
    #     # pred_y = auto_encoder.predict(list_x[0,:])
    #     pred_y[i] = auto_encoder.predict(list_x[i,:])
    #     # print(pred_y, list_y[0,:])
    #     # print(pred_y[i], list_y[i,:])

    # pred_ = np.asarray(pred_y)
    # gt_ = np.asarray(list_y)
    # MAE = np.mean(abs(pred_-gt_),axis=0)
    # print(f' The MAE is {MAE}')

if __name__ == '__main__':
    main()
