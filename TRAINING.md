
## CIFAR-100


#### ResNet-110
```
python cifar.py -a resnet -d cifar100 --depth 110 --epochs 164 --lr-decay schedule --schedule 81 122 --gamma 0.1 --wd 1e-4 -c checkpoints/cifar10/resnet-110 --mimic
```

#### WRN-28-10 (dropout 0.3)
```
python cifar.py -a wrn -d cifar100 --depth 28 --widening-factor 10 --dropout 0.3 --epochs 200 --lr-decay schedule --schedule 60 120 160 --wd 5e-4 --gamma 0.2 -c checkpoints/cifar10/wrn-28-10-drop --mimic
```

#### DenseNet-BC (L=100, k=12)
```
python cifar.py -a densenet -d cifar100 --depth 100 --growth-rate 12 --train-batch 64 --epochs 300 --lr-decay schedule --schedule 150 225 --wd 1e-4 --gamma 0.1 -c checkpoints/cifar10/densenet-bc-100-12 --mimic
```



## ImageNet

### ResNet-152
```
python imagenet.py -a resnet152 -d /path/to/ILSVRC2012/data --epochs 90 --lr-decay schedule --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet152 --mimic
```


The argument `mimic` facilitates training with our **Dynamic Hirarchical Mimicking** strategy, otherwise merely the standard deep supervision is applied to the networks.
