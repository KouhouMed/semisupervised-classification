### Results
These are the top-1 accuracy of linear classifiers trained on the 
(frozen) representations learned by SimCLR:

**SimCLR training:**

Dataset: CIFAR-10
Architecture: ResNet-18
Batch Size: 256
Epochs: 20
Project output dimensionality: 64
Optimizer: Adam
Temperature: 0.5
Weight decay': 1e-06
Learning rate: 3e-4

**Linear evaluation:**

Epochs: 500
Batch Size: 32
Learning rate: 3e-4

**Supervised:**

Batch Size: 32
Epochs: 500
Learning rate: 3e-4

| Label fraction | SimCLR | Supervised |
|----------------|--------|------------|
| 1%             | 60.7%  | 49.1%      |
| 10%            | 70.4%  |            |
| 50%            | 73.1%  | 72.1%      |
| 100%           | 74.8%  |            |


### Autoencoder with data aug (3convs + 3deconvs)
lr 1e-3
bs 32
100 epochs
Training complete in 49m 1s
Minimum test Loss: 0.551848

**evaluation**
1%
Training complete in 1m 48s
Best test Accuracy: 0.300500

10%
Training complete in 3m 17s
Best test Accuracy: 0.358200

50%
Training complete in 8m 25s
Best test Accuracy: 0.393900

100%
Training complete in 14m 16s
Best test Accuracy: 0.400300


### DAE (3convs + 3deconvs)
lr 1e-3
bs 32
100 epochs
Training complete in 36m 8s
Minimum test Loss: 0.555994

**evaluation**
1%
Training complete in 1m 44s
Best test Accuracy: 0.322600

10%
Training complete in 3m 14s
Best test Accuracy: 0.376400

50%
Training complete in 9m 5s
Best test Accuracy: 0.406900


100%
Training complete in 14m 7s
Best test Accuracy: 0.427000


### resnet AE
lr 1e-3
bs 32
100 epochs
Training complete in 83m 59s
Minimum test Loss: 0.562117

**evaluation**
1%
Training complete in 1m 37s
Best test Accuracy:  0.291600

10%
Training complete in 2m 57s
Best test Accuracy: 0.414400

50%
Training complete in 8m 53s
Best test Accuracy: 0.440000

100%
Training complete in 13m 48s
Best test Accuracy: 0.440800


### resnet DAE
Training complete in 86m 53s
Minimum test Loss: 0.563424

**evaluation**
1%
Training complete in 1m 46s
Best test Accuracy: 0.288600

10%
Training complete in 3m 3s
Best test Accuracy: 0.409500

50%
Training complete in 9m 4s
Best test Accuracy: 0.434400

100%
Training complete in 14m 15s
Best test Accuracy: 0.441400


### supervised
1%
Training complete in 6m 9s
Best test Accuracy: 0.396300

10%
Training complete in 11m 15s
Best test Accuracy: 0.647500

50%
Training complete in 37m 45s
Best test Accuracy: 0.782300

100%
Training complete in 70m 8s
Best test Accuracy: 0.823500



### SIMCLR 

*crop + crop*
1% 34.18%
10% 41.44%
50% 44.45%
100% 45.87%

