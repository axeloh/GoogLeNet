# GoogLeNet/InceptionNet for CIFAR10/100

``` 
$ git clone https://github.com/axeloh/GoogLeNet.git
$ cd GoogLeNet
```

``` 
$ python train.py
```

With other than default params:
``` 
$ python train.py --dataset cifar100 --n_epochs 100 --batch_size 128 --lr 5e-3 --use_cuda False
```

Model saved every third epoch in ``` models/ ```.
Loss and accuracy for train set and validation set saved in ``` output/ ```.

