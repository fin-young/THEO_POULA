
# THEO_POULA

This repository is the official implementation of "Polygonal Unadjusted Langevin Algorithms: Creating stable and efficient adaptive algorithms for
neural networks" (https://arxiv.org/abs/2105.13937). 


## Dependencies

- Python 3.6
- Pytorch 1.8.0 + cuda
- scikit-learn

## Training

To train the models in the paper, run the following commands:

### CNN (CIFAR10)
```train
python train_cifar.py --lr 1e-2 --eps 1e-4 --beta 1e14 --model 'VGG11' --optimizer_name='THEOPOULA' --num_epoch=200 
python train_cifar.py --lr 5e-4 --model 'VGG11' --optimizer_name='ADAM' --num_epoch=200
python train_cifar.py --lr 1e-3 --model 'VGG11' --optimizer_name='AMSGrad' --num_epoch=200 
python train_cifar.py --lr 5e-4 --model 'VGG11' --optimizer_name='RMSProp' --num_epoch=200 
```

### LSTM (Penn Treebank)
```train
python train_lstm.py --lr 5e-1 --eps 1e-1 --beta 1e10 --optimizer_name='THEOPOULA' 
python train_lstm.py --lr 1e-3 --optimizer_name='ADAM' 
python train_lstm.py --lr 5e-4 --optimizer_name='AMSGrad' 
python train_lstm.py --lr 1e-3 --optimizer_name='RMSProp' 
```

### DNN (Insurance claim)
```train
python train_gammareg.py --lr 5e-4 --eps 1e-2 --beta 1e10 --optimizer_name='THEOPOULA' --epochs=300 
python train_gammareg.py --lr 1e-4 --optimizer_name='ADAM' --epochs=300 
python train_gammareg.py --lr 5e-3 --optimizer_name='AMSGrad' --epochs=300 
python train_gammareg.py --lr 1e-4 --optimizer_name='RMSProp' --epochs=300 
```

Note that trained models are saved in "./ckpt_{model_name}/{experiment_name}/{experment_name}.pth". 
Log files and a dataframe for training and test losses are saved in  "./log_{model_name}/{experiment_name}/{log_name}" and "./log_{model_name}/{experiment_name}/history.pkl". 

## Evaluation and Plot

To evaluate and plot the models, run "plots_cifar.py", "plots_lstm.py", "plots_smodel.py". Note that you need to specify paths of log files you are interested in. 


## Results

Our model achieves the following performance on :

### CIFAR10

| Optimizer   | test loss  |               hyper-parameters                    |
| ------------|----------- | ------------------------------------------------- |
| THEOPOULA   |     0.0197 |  $(\lambda, \epsilon, \beta) = (1e-2, 1e-4, 1e14)$|




