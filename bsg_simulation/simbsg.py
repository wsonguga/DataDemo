from utils import *

if __name__ == "__main__":
    print('Generating ABP:')
    sig = signal_simulate(
        num_rows = 1000,
        heart_rate = (50,100),
        add_respiratory = False,
        respiratory_rate = (10,30),
        MAP = (50,100),
        TPR= 0.01,
        TPR_noise = 0.001,
        change_with_time = True,
        change_with_time_amp = 0.4,
        random_state = None,
        silent = False,
        data_file = "./data/dataSV_all.npy"
    )

    dataSV_all = np.load('./data/dataSV_all.npy')
    dataWV_all = np.zeros(1006).reshape(1,-1)

    print('Generating Waveform:')
    for i in tqdm(range(dataSV_all.shape[0])):
        signal_SV = dataSV_all[i,:1000]
        signal_SV_inverted = 0.4 * -signal_SV

        waveform = []
        for j in range(0,len(signal_SV_inverted),8):
            if j // 8 % 2 == 0:
                waveform.append(signal_SV_inverted[j])
            else:
                waveform.append(signal_SV[j])

        interpolation_function = interp1d(np.arange(0,1000,8), waveform, kind='quadratic')
        x_new = np.linspace(0, 992, 1000)
        y_resampled = interpolation_function(x_new)
        waveform = np.concatenate((y_resampled, dataSV_all[i,1000:]))
        dataWV_all = np.vstack((dataWV_all,waveform))

    dataWV_all = dataWV_all[1:]

    np.save("./data/dataWV_all.npy", dataWV_all)

    dataWV_all_ = dataWV_all.copy()
    dataWV_all_rr = add_respiration(dataWV_all_)
    np.save("./data/dataWV_all_rr.npy",dataWV_all_rr)

    data_train = dataWV_all_rr[:int(dataWV_all_rr.shape[0]*0.8)]
    np.save("./data/data_train.npy", data_train)
    dataWV_test_rr = dataWV_all_rr[int(dataWV_all_rr.shape[0]*0.8):]
    dataSV_test = dataSV_all[int(dataWV_all_rr.shape[0]*0.8):]

    data_test = np.ones((1,1006))
    print('Adding noise to test set:')
    for i in tqdm(range(dataWV_test_rr.shape[0])):
        sig_sv = dataSV_test[i,:1000]
        sig_wv = dataWV_test_rr[i,:1000]
        peaks, _ = find_peaks(-sig_sv)
        peaks = np.append(peaks,999)
        sig_wv_ = sig_wv.copy()
        sig_wv_new = white_noise(sig_wv_, peaks, 1.5, 10, 15)
        sig_wv_new = np.concatenate([sig_wv_new,dataSV_test[i,-6:]])
        data_test = np.vstack((data_test,sig_wv_new))
    data_test = data_test[1:]
    np.save("./data/data_test.npy",data_test)