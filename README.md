# GoogLeNet/InceptionNet for CIFAR10/100
Pytorch implementation from scratch.

``` 
$ git clone https://github.com/axeloh/GoogLeNet.git
$ cd GoogLeNet
```

``` 
$ python train.py
```

Example of train with other than default params:
``` 
$ python train.py --dataset cifar100 --epochs 100 --bs 128 --lr 5e-3 --gpu False --modelname mymodel --save_every 10 --lr_scheduler False
```
(GPU strongly recommended!)


Model saved in ``` models/ ```, loss and accuracy plot for train and validation set saved in ``` output/ ```.


### Model trained on CIFAR100
#### Without data augmentation and learning rate scheduler:

Loss | Accuracy
:--- | :---
![Alt text](/output/loss_plot.png?raw=true) | ![Alt text](/output/acc_plot.png?raw=true)


#### With data augmentation and with a step learning rate scheduler:
Loss | Accuracy
:--- | :---
![Alt text](/output/model_augmented_loss_plot.png?raw=true) | ![Alt text](/output/model_augmented_acc_plot.png?raw=true)

