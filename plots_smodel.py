import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = './log_gammareg/best/ADAM_bs{128}_hs{50}_lr{1.0e-04}_epoch{300}_eta{0.0e+00}_beta{1.0e+10}_eps{1.0e-02}/history.pkl'
df1 = pkl.load(open(path, 'rb'))

path = './log_gammareg/best/AMSGrad_bs{128}_hs{50}_lr{5.0e-03}_epoch{300}_eta{0.0e+00}_beta{1.0e+10}_eps{1.0e-02}/history.pkl'
df2 = pkl.load(open(path, 'rb'))

path = './log_gammareg/best/RMSProp_bs{128}_hs{50}_lr{1.0e-04}_epoch{300}_eta{0.0e+00}_beta{1.0e+10}_eps{1.0e-02}/history.pkl'
df3 = pkl.load(open(path, 'rb'))

path = './log_gammareg/best/THEOPOULA_bs{128}_hs{50}_lr{5.0e-04}_epoch{300}_eta{0.0e+00}_beta{1.0e+10}_eps{1.0e-02}/history.pkl'
df4 = pkl.load(open(path, 'rb'))


plt.figure(1)


for i, key in zip(range(1, 5), df1.keys()):
    plt.figure(i)
    plt.plot(df1[key], label='ADAM')
    plt.plot(df2[key], label='AMSGRAD')
    plt.plot(df3[key], label='RMSProp')
    plt.plot(df4[key], label='TheoPoula')
    plt.legend()
    plt.title(key)
    if i==1:
        plt.ylim([8.2, 11])
    else:
        plt.ylim([8.56, 8.9])

plt.show()
