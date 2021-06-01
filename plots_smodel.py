import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = './log_gammareg/best/ADAM_bs{128}_hs{50}_lr{1.0e-04}_epoch{300}_eta{0.0e+00}_beta{1.0e+10}_eps{1.0e-02}/history.pkl'
df_adam = pkl.load(open(path, 'rb'))

path = './log_gammareg/best/AMSGrad_bs{128}_hs{50}_lr{5.0e-03}_epoch{300}_eta{0.0e+00}_beta{1.0e+10}_eps{1.0e-02}/history.pkl'
df_ams = pkl.load(open(path, 'rb'))

path = './log_gammareg/best/RMSProp_bs{128}_hs{50}_lr{1.0e-04}_epoch{300}_eta{0.0e+00}_beta{1.0e+10}_eps{1.0e-02}/history.pkl'
df_rms = pkl.load(open(path, 'rb'))

path = './log_gammareg/best/THEOPOULA_bs{128}_hs{50}_lr{5.0e-04}_epoch{300}_eta{0.0e+00}_beta{1.0e+10}_eps{1.0e-02}/history.pkl'
df_theopoula = pkl.load(open(path, 'rb'))


plt.figure(1)


for i, key in zip(range(1, 5), df_adam.keys()):
    plt.figure(i)
    plt.plot(df_adam[key], label='ADAM')
    plt.plot(df_ams[key], label='AMSGRAD')
    plt.plot(df_rms[key], label='RMSProp')
    plt.plot(df_theopoula[key], label='TheoPoula')
    plt.legend()
    plt.title(key)
    if i==1:
        plt.ylim([8.2, 11])
    else:
        plt.ylim([8.56, 8.9])

plt.show()

print('AMSGrad:  test_nll -',np.array(df_ams['test_nll']).min().item())
print('ADAM:  test_nll -',np.array(df_adam['test_nll']).min().item())
print('RMSprop:  test_nll -',np.array(df_rms['test_nll']).min().item())
print('THEOPOULA:  test_nll -',np.array(df_theopoula['test_nll']).min().item())