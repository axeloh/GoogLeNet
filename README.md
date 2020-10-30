# GoogLeNet/InceptionNet for CIFAR10/100
Pytorch implementation from scratch.

``` 
$ git clone https://github.com/axeloh/GoogLeNet.git
$ cd GoogLeNet
```

``` 
$ python train.py
```

With other than default params:
``` 
$ python train.py --dataset cifar100 --epochs 100 --bs 128 --lr 5e-3 --gpu False --modelname mymodel --save_every 5 --lr_scheduler True
```

Model saved every third epoch in ``` models/ ```.
Loss and accuracy for train set and validation set saved in ``` output/ ```.


### Loss during training 
![Alt text](/output/loss_plot.png?raw=true)

### Accuracy during training 
![Alt text](/output/acc_plot.png?raw=true)
