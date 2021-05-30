## package
import torch
import torch.optim as optim
import os
import argparse
import numpy as np
import pickle as pkl
import pandas as pd
from utils import get_nll_S
from models_gamma import S_net
import matplotlib.pyplot as plt
from custom_optim import THEOPOULA
from utils import FS_Dataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

## parsing
parser = argparse.ArgumentParser('Nonlinear gamma regression')
parser.add_argument('--s_dist', default='Gamma', type=str, help='severity distribution')

parser.add_argument('--trial', default='trial1', type=str)
parser.add_argument('--bs', default=128, type=int, help='batch_size')
parser.add_argument('--epochs', default=300, type=int, help='# of epochs')
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--act_fn', default='elu', type=str)
parser.add_argument('--hs', default=50, type=int, help='number of neurons')
parser.add_argument('--optimizer_name', default='AMSGrad', type=str)
parser.add_argument('--with_BN', action='store_true') #do not use Batch normalization
parser.add_argument('--with_DO', action='store_true') #do not use Dropout
parser.add_argument('--p_dropout', default=0.25, type=float)
parser.add_argument('--eta', default='0', type=float)
parser.add_argument('--beta', default='1e10', type=float)
parser.add_argument('--eps', default=1e-2, type=float)

parser.add_argument('--log_dir', default='./log_gammareg/', type=str)
parser.add_argument('--ckpt_dir', default='./ckpt_gammareg/', type=str)


torch.manual_seed(1111)

args = parser.parse_args()

s_dist = args.s_dist
trial = args.trial
batch_size = args.bs
epochs = args.epochs

eta = args.eta
beta = args.beta
eps = args.eps

with_BN = args.with_BN
with_DO = args.with_DO

p_dropout = args.p_dropout
lr = args.lr
act_fn = args.act_fn
hidden_size = args.hs
optimizer_name = args.optimizer_name


log_dir = args.log_dir


device = 'cuda' if torch.cuda.is_available() else 'cpu'
## data preparation
print(with_BN, with_DO)
print('==> Preparing data..')

file = open('./data_insurance/refined_data_category.pkl', 'rb')
data = pkl.load(file)

num_features = (data.shape[1] - 3)

train_data, test_data = train_test_split(data, random_state=0, test_size=0.3)

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

print(train_data.shape, train_data['freq'].value_counts())
print(test_data.shape, test_data['freq'].value_counts())

trainloader = torch.utils.data.DataLoader(FS_Dataset(train_data), batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(FS_Dataset(test_data), batch_size=batch_size, shuffle=False)

print('==> train data..shape: ', train_data.shape)
print(train_data.head())
print('==> test data..shape: ', test_data.shape)
print(test_data.head())


num_data = len(train_data)
num_batch = np.ceil(num_data / batch_size)

experiment_name = '%s_bs{%d}_hs{%d}_lr{%.1e}_epoch{%d}_eta{%.1e}_beta{%.1e}_eps{%.1e}' \
                      %(optimizer_name, batch_size, hidden_size, lr, epochs, eta, beta, eps)

log_dir = args.log_dir + experiment_name
ckpt_dir = args.ckpt_dir + experiment_name

writer = SummaryWriter(log_dir=log_dir)

## Frequency model, optimizer
print('==> Start severity model..')

## Preparing data and dataloader
print('==> Preparing data..')

train_data = train_data.loc[train_data['freq']>0].reset_index(drop=True)
test_data = test_data.loc[test_data['freq']>0].reset_index(drop=True)

trainloader = torch.utils.data.DataLoader(FS_Dataset(train_data), batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(FS_Dataset(test_data), batch_size=batch_size, shuffle=False)

best_nll = 999
state = []
best_epoch = 0
num_data = len(train_data)
num_batch = np.ceil(num_data/batch_size)


## Severity model, optimizer

print('==> Building model.. on {%s}'%device)

S_model = S_net(input_size=num_features,
                hidden_size=hidden_size,
                act_fn=act_fn,
                with_BN=with_BN,
                with_DO=with_DO,
                p_dropout=p_dropout
                )
S_model.to(device)

print('initial gamma %.4f'%(S_model.l4.weight.data))
print('==> Set optimizer.. use {%s}'%optimizer_name)
if optimizer_name == 'SGD':
    optimizer = optim.SGD(S_model.parameters(), lr=lr)
elif optimizer_name =='ADAM':
    optimizer = optim.Adam(S_model.parameters(), lr=lr)
elif optimizer_name == 'RMSProp':
    optimizer = optim.RMSprop(S_model.parameters(), lr=lr)
elif optimizer_name == 'AMSGrad':
    optimizer = optim.Adam(S_model.parameters(), lr=lr, amsgrad=True)
elif optimizer_name == 'THEOPOULA':
    optimizer = THEOPOULA(S_model.parameters(), lr=lr, eta=eta, beta=args.beta, eps=eps)

history = {'training_nll': [],
           'test_nll': [],
           }

## Training - severity model
print('==> Start training - Severity model')
loss_mse = torch.nn.MSELoss()
hist_train_nll = []
hist_test_nll = []
hist_phi = []
hist_gamma = []
state = {}

def S_train(epoch, net):
    global hist_train_nll, hist_phi, hist_gamma
    print('\n Epoch: %d'%epoch)
    net.train()
    train_nll = []
    for batch_idx, samples in enumerate(trainloader):
        samples = samples.to(device)
        optimizer.zero_grad()

        cov = samples[:, :num_features]
        y = samples[:, -1]
        output = net(cov)

        if (epoch == 1) & (batch_idx == 1):
            hist_phi += [output[1].item()]

        nll = get_nll_S(s_dist, y, output)
        nll.backward()
        optimizer.step()

        train_nll += [nll.item()]
        if batch_idx % 200 ==0:
            print('TRAIN: EPOCH %04d/%04d | BATCH %04d/%04d | NLL %.4f | Dispersion %.4f'
              %(epoch, epochs, batch_idx, num_batch, np.mean(train_nll), output[1].cpu().data))

    print('TRAIN: EPOCH %04d/%04d | NLL %.4f | Dispersion %.4f'
          % (epoch, epochs, np.mean(train_nll), output[1].cpu().data))
    writer.add_scalar('S_Training nll', np.mean(train_nll), epoch)

    hist_train_nll += [np.mean(train_nll)]
    hist_phi += [output[1].item()]

    history['training_nll'].append(np.mean(train_nll))

def S_test(epoch, net):
    global state, best_nll, hist_test_nll, best_epoch
    net.eval()
    test_nll =[]

    with torch.no_grad():
        for batch_idx, samples in enumerate(testloader):
            samples = samples.to(device)

            cov = samples[:, :num_features]
            y = samples[:, -1]

            output = net(cov)
            nll = get_nll_S(s_dist, y, output)

            test_nll += [nll.item()]

        print('TEST: NLL %.4f '% (np.mean(test_nll)))


    if np.mean(test_nll) < best_nll:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': np.mean(test_nll),
            'epoch': epoch,
            'optim': optimizer.state_dict()
        }
        best_nll = np.mean(test_nll)

    writer.add_scalar('S_Test nll', np.mean(test_nll), epoch)
    hist_test_nll += [np.mean(test_nll)]
    history['test_nll'].append(np.mean(test_nll))

for epoch in range(1, epochs+1):
    S_train(epoch, S_model)
    S_test(epoch, S_model)

plt.figure(1)
plt.plot(range(2, epochs+1), hist_train_nll[1:], label='train')
plt.plot(range(2, epochs+1), hist_test_nll[1:], label='test')
plt.xlabel('epochs')
plt.ylabel('nll')
plt.legend()

#save result
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
pkl.dump(history, open(log_dir+'/history.pkl', 'wb'))

# save model
if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)
torch.save(state, './%s/%s.pth' % (ckpt_dir, experiment_name))


plt.show()

