from utils import *

if __name__ == "__main__":
    print('Start to Generate ABP')
    # The range of TPR: 0.005-0.006 is train; 0.003-0.004 is test.
    ABP_train = []
    ABP_test = []

    for i in tqdm(range(55, 96)):
        sig = signal_simulate(
            num_rows = 1000,
            heart_rate = (50,100),
            MAP = (i, i),
            TPR = (0.005, 0.006),
            random_state = None,
            silent = False,
            data_file = "./data/ABP_train.npy"
        )
        sig = np.load("./data/ABP_train.npy")
        ABP_train.append(sig)

    for i in tqdm(range(55, 96)):
        sig = signal_simulate(
            num_rows = 250,
            heart_rate = (50,100),
            MAP = (i, i),
            TPR = (0.003, 0.004),
            random_state = None,
            silent = False,
            data_file = "./data/ABP_test.npy"
        )
        sig = np.load("./data/ABP_test.npy")
        ABP_test.append(sig)
    print('ABP Generation Done')

    ABP_train = np.vstack(ABP_train)
    ABP_test = np.vstack(ABP_test)
    np.random.shuffle(ABP_train)
    np.random.shuffle(ABP_test)
    ABP_train = ABP_train[:40000]
    ABP_test = ABP_test[:10000]
    print('The Shape of ABP_train is:', ABP_train.shape)
    print('The Shape of ABP_test is:', ABP_test.shape)
    # print(np.min(data_train[:,-1]), np.max(data_train[:,-1]))
    # print(np.min(data_test[:,-1]), np.max(data_test[:,-1]))
    np.save("./data/ABP_train.npy", ABP_train)
    np.save("./data/ABP_test.npy", ABP_test)

    print("Start to Generate BSG based on ABP")
    BSG_train = []
    BSG_test = []
    for i in tqdm(range(ABP_train.shape[0])):
        BSG_tmp = generate_BSG_from_ABP(ABP_train[i, :1000])
        BSG_tmp = np.concatenate((BSG_tmp, ABP_train[i, -6:]))
        BSG_train.append(BSG_tmp)
    for i in tqdm(range(ABP_test.shape[0])):
        BSG_tmp = generate_BSG_from_ABP(ABP_test[i, :1000])
        BSG_tmp = np.concatenate((BSG_tmp, ABP_test[i, -6:]))
        BSG_test.append(BSG_tmp)
    BSG_train = np.vstack(BSG_train)
    BSG_test = np.vstack(BSG_test)
    # print(BSG_train.shape)
    # print(BSG_test.shape)
    np.save("./data/BSG_train.npy", BSG_train)
    np.save("./data/BSG_test.npy", BSG_test)
    print('BSG Generation Done')

    print("Start to Add respiration on BSG")
    BSG_train_ = BSG_train.copy()
    BSG_test_ = BSG_test.copy()
    BSG_train_rr = add_respiration(BSG_train_)
    BSG_test_rr = add_respiration(BSG_test_, 0.1)
    np.save("./data/BSG_train_rr.npy", BSG_train_rr)
    np.save("./data/BSG_test_rr.npy", BSG_test_rr)
    print('Respiration Added')

    print('Start to Add Noise')
    BSG_test_rr_ = BSG_test_rr.copy()
    BSG_train_rr_ = BSG_train_rr.copy()
    BSG_train_rr_noise = white_noise(BSG_train_rr_, 3, 30, 40)
    BSG_test_rr_noise = white_noise(BSG_test_rr_, 3, 30, 40)
    np.save("./data/BSG_train_rr_noise.npy", BSG_train_rr_noise)
    np.save("./data/BSG_middle_rr_noise.npy", BSG_test_rr_noise)
    print('Noise Added')