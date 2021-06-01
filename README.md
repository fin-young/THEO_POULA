
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

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
