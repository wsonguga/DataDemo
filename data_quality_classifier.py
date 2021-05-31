'''
Author: Ming Song
Data Quality Classifier
ming.song@uga.edu
CopyRight@2021 MING
'''

import os
import sys
import numpy as np
import torch
from torch.optim import *
from torch import nn, optim, cuda
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# from torch.utils.data import
from sklearn import preprocessing
import copy
from random import sample
from math import isnan
from sklearn.metrics import accuracy_score, f1_score
import datetime
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# batch_size = 1000
# test_only = False
# VISUAL_FLAG = False
# # test_only = bool(sys.argv[1])
# lr = 0.001
# dim_feature = 100
class Initial_Dataset(Dataset):
    """docstring for ."""

    def __init__(self, data, data_length):  # before this length is data, after is label
        self.data = data
        self.data_length = data_length

        self.array_Tx = []
        for each_data in self.data:
            self.array_Tx.append(each_data[:1000]/max(each_data[:1000]))
        # import pdb; pdb.set_trace()
        self.array_Tx = np.asarray(self.array_Tx)[:,np.newaxis,:]

        self.array_Ty = self.data[:,-1][:,np.newaxis,np.newaxis]
        # import pdb; pdb.set_trace()
        #
        # if scalers == None:
        #
        #     self.scaler_x = preprocessing.StandardScaler().fit(self.data[:,:self.data_length])
        #     self.scaler_x.scale_[:] = np.std(self.data[:,:self.data_length])
        #     self.scaler_x.mean_[:] = np.mean(self.data[:,:self.data_length])
        #
        #     self.array_Tx_temp = self.scaler_x.transform(self.data[:,:self.data_length])
        #     self.array_Tx_temp = self.array_Tx_temp[:, :(len(self.array_Tx_temp[0]) // self.dim_feature) * self.dim_feature]
        #     self.array_Tx = self.array_Tx_temp.reshape(len(self.array_Tx_temp),-1,self.dim_feature)
        #
        #
        #     self.scaler_y = preprocessing.StandardScaler().fit(self.data[:,-4:])
        #     # self.array_Ty = self.scaler_y.transform(self.data[:,-4:])
        #     self.array_Ty = self.data[:,-4:]
        #
        # else:
        #     self.scaler_x = scalers[0]
        #     self.scaler_y = scalers[1]
        #     self.array_Tx_temp = self.scaler_x.transform(self.data[:,:self.data_length])
        #     self.array_Tx_temp = self.array_Tx_temp[:, :(len(self.array_Tx_temp[0]) // self.dim_feature) * self.dim_feature]
        #     self.array_Tx = self.array_Tx_temp.reshape(len(self.array_Tx_temp),-1,self.dim_feature)
        #
        #     # self.array_Ty = self.scaler_y.transform(self.data[:,-4:])
        #     self.array_Ty = self.data[:,-4:]


    def __getitem__(self, index):
        data_ = self.array_Tx[index, :]
        gt_ = self.array_Ty[index, :] #

        return data_, gt_

    # def get_scalers(self):
    #     return self.scaler_x, self.scaler_y

    def __len__(self):
        return self.data.shape[0]


class LstmAttentionNet(nn.Module):
    def __init__(self):
        super(LstmAttentionNet, self).__init__()
        hidden_size = 6
        attention_size = hidden_size
        self.lstm = nn.LSTM(input_size=1000, hidden_size=hidden_size, batch_first=True, num_layers=2)
        self.w_omega = nn.Parameter(torch.randn(hidden_size,attention_size))
        self.b_omega = nn.Parameter(torch.randn(attention_size))
        self.u_omega = nn.Parameter(torch.randn(attention_size,1))
        self.decoding_layer = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        v = torch.matmul(out,self.w_omega)+self.b_omega
        vu = torch.matmul(v, self.u_omega)
        weight= nn.functional.softmax(vu,dim=1)
        out_weighted = torch.sum(out*weight,1)
        y_pred = self.decoding_layer(out_weighted)

        return y_pred#, weight



class CNN_classification(nn.Module):
    def __init__(self):
        super(CNN_classification, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=7)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7904, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # import pdb; pdb.set_trace()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        # import pdb; pdb.set_trace()
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        # output = F.softmax(x, dim=1)
        # import pdb; pdb.set_trace()
        return x


class DATA_QUALITY_Model():
    """docstring for DATA_QUALITY_Model."""

    def __init__(self, LOG=False):
        super(DATA_QUALITY_Model, self).__init__()
        ####
        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')

        # self.ae_Net = AE()
        self.Net = CNN_classification()
        # self.Net = LstmAttentionNet()
        # self.ae_Net = Batch_Net_2(3000,256,128,64,32)
        # self.ae_Net = Batch_Net(3000,256,128)
        # self.ae_Net = self.ae_Net.to(device = self.device)
        self.Net = self.Net.to(device = self.device)
        # self.ae_Net = nn.DataParallel(self.ae_Net,device_ids=[0,1,2,3]) # multi-GPU
        self.Net = nn.DataParallel(self.Net,device_ids=[0,1,2,3]) # multi-GPU
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
        # class_num = 55
        # organize data to trainable form
        organized_data = self.organize_data_from_npy(self.data)
        # organized_data = self.data
        # devide train and test data set
        devide_index = int(organized_data.shape[0]*devide_factor)

        # get train and test dataset
        data_train = organized_data[:devide_index,:]
        data_test = organized_data[devide_index:,:]

        train_dataset = Initial_Dataset(data_train, self.window_len)
        test_dataset = Initial_Dataset(data_test, self.window_len)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        ### training component
        # loss_fn_AE = torch.nn.MSELoss()
        loss_fn_CNN = torch.nn.CrossEntropyLoss()
        # optimizer_AE = optim.Adam(self.ae_Net.parameters(), lr=learning_rate)
        optimizer_CNN = optim.Adam(self.Net.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer_CNN,step_size=15, gamma = 0.95)

        self.last_error = 1e5
        for e in range(epoch_number):
            # print(e)
            # lr = optimizer_AE.state_dict()['param_groups'][0]['lr']
            # print(f'Learning Rate: {lr}')

            for train_tensor_x, train_tensor_y in train_loader:
                # import pdb; pdb.set_trace()

                train_tensor_x = torch.tensor(train_tensor_x,dtype=torch.float32,device=self.device)
                train_tensor_y = torch.tensor(train_tensor_y,dtype=torch.float32,device=self.device)

                train_y_pred_cnn = self.Net(train_tensor_x)
                # import pdb; pdb.set_trace()
                train_loss_tensor_CNN = loss_fn_CNN(train_y_pred_cnn, torch.squeeze(train_tensor_y).long())

                train_loss = train_loss_tensor_CNN.item()

                optimizer_CNN.zero_grad()
                train_loss_tensor_CNN.backward()
                optimizer_CNN.step()

                # import pdb; pdb.set_trace()

            print(f'Epoch {e} train MSE: {train_loss} ')
            print(f'      --> train MSE of Classification Net: {train_loss} ')


            #
            if e % 1 == 0:
                loss_test_cnn = []
                for test_tensor_x, test_tensor_y in test_loader:
                    test_tensor_x = torch.tensor(test_tensor_x,dtype=torch.float32,device=self.device)
                    test_tensor_y = torch.tensor(test_tensor_y,dtype=torch.float32,device=self.device)

                    # import pdb; pdb.set_trace()

                    test_y_pred_cnn = self.Net(test_tensor_x)
                    # import pdb; pdb.set_trace()
                    test_loss_tensor_CNN = loss_fn_CNN(test_y_pred_cnn, torch.squeeze(test_tensor_y).long().unsqueeze(0))
                    test_loss_CNN = test_loss_tensor_CNN.item()
                    loss_test_cnn.append(test_loss_CNN)

                print(f'Epoch {e} test CE: {np.mean(loss_test_cnn)} ')
                self.error = np.mean(loss_test_cnn)
                if self.error < self.last_error:

                    if not os.path.exists('../Data_Quality_Classifier_models'):
                        os.makedirs('../Data_Quality_Classifier_models')
                        print(f'make dir ../Data_Quality_Classifier_models')


                    self.save_model(model_path='../Data_Quality_Classifier_models')
                    self.last_error = self.error


            # learning rate decay
            scheduler.step()
            print('--------------------------------------------------------------')
                # import pdb; pdb.set_trace()


        # import pdb; pdb.set_trace()

    def save_model(self, model_path='../models'):

        print('save model...')
        # with open(os.path.join(model_path,"scaler_param.pk"),"wb+") as f:
        #     pickle.dump([self.scaler_x,self.scaler_y,self.window_len],f)

        torch.save(self.Net.state_dict(), os.path.join(model_path,"cnn_model_param.pk"))

        print('save done!')


    def load_model(self, model_path='../models'):
        # if os.path.exists(os.path.join(model_path,"scaler_param.pk")):
        #     with open(os.path.join(model_path,"scaler_param.pk"),"rb+") as f:
        #         [self.scaler_x,self.scaler_y,self.window_len] = pickle.load(f)
        # else:
        #     print(f'scaler_param.pk not exist!')
        #     quit()

        if os.path.exists(os.path.join(model_path,"cnn_model_param.pk")) :
            self.Net.load_state_dict(torch.load(os.path.join(model_path,"cnn_model_param.pk"),map_location=torch.device(self.device)))

        else:
            print(f'model_param.pk not exist!')
            quit()

        print('Model parameters loaded!')


    def predict(self, pred_x):

        self.Net.eval()
        # self.pred_x = self.scaler_x.transform([pred_x[:3000]])
        # self.pred_x = self.pred_x[:, :(self.window_len // 100) * 100].reshape(len(self.pred_x),-1,100)
        self.pred_x = pred_x[:1000]/max(pred_x[:1000]).reshape(1,1,-1)

        # import pdb; pdb.set_trace()

        self.tensor_pred_x = torch.tensor(self.pred_x,dtype=torch.float32,device=self.device)
        # self.train_y_pred = self.scaler_y.inverse_transform(self.ae_Net(self.tensor_pred_x[:,np.newaxis,:]).cpu().detach().numpy())
        # import pdb; pdb.set_trace()
        self.train_y_pred = torch.argmax(torch.softmax(self.Net(self.tensor_pred_x),1),1).cpu().detach().numpy()

        # import pdb; pdb.set_trace()

        return self.train_y_pred


    def evaluate(self, data, VERBOSE=False, INFLUX=False):
        self.data = data
        # self.ae_Net.eval()
        self.Net.eval()
        # loss_fn_AE = torch.nn.MSELoss()
        loss_fn_CNN = torch.nn.CrossEntropyLoss()
        organized_data = self.organize_data_from_npy(self.data)
        devide_index = int(organized_data.shape[0]*0.8)

        # get train and test dataset
        data_train = organized_data[:devide_index,:]
        data_test = organized_data[devide_index:,:]

        dataset = Initial_Dataset(data_test, 1000)

        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        pred_ = []
        gt_ = []
        loss_ = []
        acc_result_ = []

        for tensor_x, tensor_y in test_loader:
            # import pdb; pdb.set_trace()

            # tensor_x = torch.reshape(tensor_x, (-1,self.window_len))
            # import pdb; pdb.set_trace()
            tensor_x = tensor_x.to(self.device)
            tensor_y = tensor_y.to(self.device)


            # y_pred = self.ae_Net(tensor_x.float()).cpu().detach().numpy().squeeze()
            # input = tensor_x.cpu().detach().numpy().squeeze()


            # test_y_pred_AE, conv2_logit_test = self.ae_Net(tensor_x.float())
            test_y_pred_cnn = self.Net(tensor_x.float())
            test_y_pred_cnn_label = torch.argmax(torch.softmax(test_y_pred_cnn,1),1)

            # import pdb; pdb.set_trace()


            # test_loss_tensor_AE = loss_fn_AE(test_y_pred_AE, tensor_x)
            # import pdb; pdb.set_trace()
            test_loss_tensor_CNN = loss_fn_CNN(test_y_pred_cnn, torch.squeeze(tensor_y).long().unsqueeze(0))
            test_loss_tensor = test_loss_tensor_CNN

            # import pdb; pdb.set_trace()

            # test_loss_AE = test_loss_tensor_AE.item()
            test_loss_CNN = test_loss_tensor_CNN.item()
            test_loss = test_loss_tensor.item()

            loss_.append(test_loss)

            pred_.append(test_y_pred_cnn_label.cpu().detach().numpy().squeeze())
            gt_.append(tensor_y.cpu().detach().numpy().squeeze())
            # import pdb; pdb.set_trace()

            # if test_y_pred_cnn_label.cpu().detach().numpy().squeeze() == tensor_y.cpu().detach().numpy().squeeze():
            #     acc_result_.append(1)
            # else:
            #     acc_result_.append(0)

            # if test_y_pred_cnn_label.cpu().detach().numpy().squeeze() == 0:
            #     print(tensor_y.cpu().detach().numpy().squeeze())
            #     plt.plot(tensor_x.squeeze())
            #     plt.show()

        # acc_result_ = np.asarray(acc_result_)
        # right_ = np.where(acc_result_[:]==1)[0].shape[0]
        # Acc = right_/acc_result_.shape[0]
        #
        # print(f'Acc: {Acc}')
        pred_ = np.asarray(pred_)
        gt_ = np.asarray(gt_)
        # import pdb; pdb.set_trace()
        Acc_ = accuracy_score(gt_, pred_)
        F1_ = f1_score(gt_, pred_)

        print(f'Acc: {Acc_}')
        print(f'F1: {F1_}')

        # import pdb; pdb.set_trace()


        #
        # plt.figure()
        # plt.plot(pred_,'b',label='pred')
        # plt.plot(gt_,'r',label='gt')
        # plt.show()


        # plt.savefig('../result.png')



def main():
    dataset = np.load('../../LabelTool/data/classifier_train_data.npy')
    # vali_dataset = np.load('../../LabelTool/data/classifier_train_data.npy')
    vali_dataset = np.load('../data/H1_0205_all_data.npy')
    # vali_dataset = np.load('../../LabelTool/data/classifier_train_data.npy')
    # np.random.shuffle(dataset)

    data_quality_model = DATA_QUALITY_Model()
    # data_quality_model.fit(all_data=dataset, window_len=1000, devide_factor=0.8,learning_rate=0.0005, batch_size=64, dim_feature=500)
    data_quality_model.load_model('../Data_Quality_Classifier_models')
    # data_quality_model.evaluate(vali_dataset)
    #
    # # pdf_good = PdfPages('../Good_data.pdf')
    # # pdf_bad = PdfPages('../Bad_data.pdf')
    good_data = []
    # # bad_data = []
    for jnd, each_vali in enumerate(vali_dataset):
        cur_pred = data_quality_model.predict(each_vali)
        # if (len(good_data) >= 40) and  (len(bad_data) >= 40):
        #     break
        if cur_pred == 0:
            good_data.append(each_vali)
            # plt.plot(each_vali[:1000])
            # plt.show()
            # pdf_good.savefig()
            # plt.cla()
        else:
            # bad_data.append(each_vali[:1000])
            # plt.plot(each_vali[:1000])
            # plt.show()
            pass
            # pdf_bad.savefig()
            # plt.cla()
    good_data = np.asarray(good_data)
    # # bad_data = np.asarray(bad_data)[0:20,:]
    import pdb; pdb.set_trace()
    #
    # # pdf_good.close()
    # # pdf_bad.close()
    # # plt.close()

if __name__ == '__main__':
    main()
