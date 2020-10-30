# GoogLeNet/InceptionNet for CIFAR10/100

Clone 
``` 
$ git clone https://github.com/axeloh/GoogLeNet.git
$ cd GoogLeNet
```

Train:
``` 
$ python train.py
```

Parameters:

- *dataset*, choices=[cifar10, cifar100] | default: cifar10
- *n_epochs*, default: 50
- *batch_size*, default: 64
- *lr*, default: 1e-3
- *use_cuda*, default: True

Train with other params:
``` 
$ python train.py --dataset cifar100 --n_epochs 100 --batch_size 128 --lr 5e-3 --use_cuda False
```

Model saved every third epoch in ``` models/ ```.
Loss and accuracy for train set and validation set saved in ``` output/ ```.

