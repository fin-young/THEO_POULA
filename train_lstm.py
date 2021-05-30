import argparse
import time
import math
import torch
import torch.nn as nn
import corpus
import model_lstm
import numpy as np
import torch.optim as optim
from custom_optim import THEOPOULA
import pickle as pkl
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')

# parser.add_argument('--opt', type=str,  default='ADAM',
#                     help='SGD, Adam, RMSprop, Momentum')


parser = argparse.ArgumentParser(description = 'pytorch PTB')
parser.add_argument('--trial', default='trial1', type=str)
parser.add_argument('--data', type=str, default='./input', # /input
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=300)
parser.add_argument('--hidden_size', default=300, type=int)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--bptt', type=int, default=20)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--optimizer_name', default='THEOPOULA', type=str)
parser.add_argument('--lr', default=5e-1, type=float, help='learning rate')

parser.add_argument('--eta', default='0', type=float)
parser.add_argument('--beta', default='1e10', type=float)
parser.add_argument('--r', default=5, type=int)
parser.add_argument('--eps', default=1e-1, type=float)

parser.add_argument('--log_dir', type=str,  default='./log_ptb/')
parser.add_argument('--ckpt_dir', default='./ckpt_ptb/', type=str)



args = parser.parse_args()

torch.manual_seed(1111)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# Load data
corpus = corpus.Corpus(args.data)

history = {'training_loss': [],
           'test_loss': [],
           'training_ppl': [],
           'test_ppl': [],
           }
trial = args.trial
optimizer_name = args.optimizer_name
batch_size = args.batch_size
hidden_size = args.hidden_size
lr = args.lr
num_epoch = args.num_epoch
eta = args.eta
beta = args.beta
r = args.r
eps = args.eps




experiment_name = '%s_bs{%d}_hs{%d}_lr{%.1e}_epoch{%d}_beta{%.1e}_r{%d}_eps{%.1e}' \
                      %(optimizer_name, batch_size, hidden_size, lr, num_epoch, beta, r, eps)
log_dir = args.log_dir + experiment_name
ckpt_dir = args.ckpt_dir + experiment_name



best_loss = 999
state = []

writer = SummaryWriter(log_dir=log_dir)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size) # size(total_len//bsz, bsz)
val_data = batchify(corpus.test, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# Build the model
num_batch = 1000 # interval to report
ntokens = len(corpus.dictionary) # 10000
model = model_lstm.RNNModel(ntokens, args.emsize, args.hidden_size, args.nlayers, args.dropout).to(device)



print(model)
criterion = nn.CrossEntropyLoss()

# Training

def repackage_hidden(h):
    return tuple(v.clone().detach() for v in h)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].clone().detach()
    target = source[i+1:i+1+seq_len].clone().detach().view(-1)

    return data, target


def evaluate(data_source):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(eval_batch_size) #hidden size(nlayers, bsz, hdsize)
        for i in range(0, data_source.size(0) - 1, args.bptt):# iterate over every timestep
            data, targets = get_batch(data_source, i)
            data, targets= data.to(device), targets.to(device)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(output, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss / len(data_source)


def train():

    model.train()
    train_loss = []
    ppl = []
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        data, targets = data.to(device), targets.to(device)

        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        opt.zero_grad()
        loss.backward()

        opt.step()
        train_loss += [loss.item()]
        ppl +=[math.exp(loss.item())]


        if batch % 1000 == 0:
            print('TRAIN: EPOCH %04d/%04d | BATCH %04d/%04d | LOSS: %.4f |  PPL %.4f' %
                      (epoch, args.num_epoch, batch, num_batch, train_loss[-1], ppl[-1]))
    print('TRAIN: EPOCH %04d/%04d | LOSS: %.4f |  PPL %.4f' %
          (epoch, args.num_epoch, np.mean(train_loss), np.mean(ppl)))

    writer.add_scalar('training_loss', np.mean(train_loss), epoch)
    writer.add_scalar('training_ppl', np.mean(ppl), epoch)

    history['training_loss'].append(np.mean(train_loss))
    history['training_ppl'].append(np.mean(ppl))

lr = args.lr
best_val_loss = None

if optimizer_name == 'SGD':
    opt = optim.SGD(model.parameters(), lr=lr)
elif optimizer_name == 'RMSProp':
    opt = optim.RMSprop(model.parameters(), lr=lr)
elif optimizer_name == 'ADAM':
    opt = optim.Adam(model.parameters(), lr=lr)
elif optimizer_name == 'AMSGrad':
    opt = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
elif optimizer_name == 'THEOPOULA':
    opt = THEOPOULA(model.parameters(), lr=lr, eta=eta, beta=args.beta, r=r, eps=eps)




try:
    for epoch in range(1, args.num_epoch+1):
        epoch_start_time = time.time()
        train()
        test_loss = evaluate(test_data)
        print('-' * 89)
        print('TEST:  LOSS: %.4f |  PPL %.4f' %
                  (test_loss,  math.exp(test_loss)))

        print('-' * 89)

        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_ppl', math.exp(test_loss), epoch)

        history['test_loss'].append(test_loss)
        history['test_ppl'].append(math.exp(test_loss))

        if test_loss < best_loss:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': test_loss,
                'epoch': epoch,
                'optim': opt.state_dict()
            }
            best_loss = test_loss

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

plt.figure(1)
plt.plot(range(1, num_epoch+1), history['training_ppl'], label='train')
plt.plot(range(1, num_epoch+1), history['test_ppl'], label='test')
plt.xlabel('epochs')
plt.ylabel('ppl')
plt.legend()
plt.show()

if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
pkl.dump(history, open(log_dir+'/history.pkl', 'wb'))


if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)
torch.save(state, './%s/%s.pth' % (ckpt_dir, experiment_name))
