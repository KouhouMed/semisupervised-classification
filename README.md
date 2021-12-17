# Semi-Supervised Image Classification

## Requirements
```
pip install simclr
```
Our scripts use the contrastive loss function NT-Xent implemented in this package.

## Scripts

`cifar10_supervised.py`
This script concerns the supervised baseline with ResNet18.

Example of usage:

```
python3 cifar10_supervised.py --data_dir data -lr 1e-3 --num_epochs 100 -bs 32 > best_accuracies.txt
```

Such command will train the supervised on 4 times, on 1%, 10%, 50%, 100% of train set 
and print out (in `best_accuracies.txy`) the best test accuracy for every phase.

`cifar10_autoencoder.py`:
This script concerns the autoencoder for reconstruction and denoising baselines.

Examples of usage:

```
python3 cifar10_autoencoder.py --data_dir data --model_name vanilla_ae -lr 1e-3 --num_epochs 20 -bs 32
python3 cifar10_autoencoder.py --model_name vanilla_dae --save_dir models
python3 cifar10_autoencoder.py --model_name resnet_ae
python3 cifar10_autoencoder.py --model_name resnet_dae
```

Suchs commands will train the autoencoder (in an unsupervised manner) on cifar10 train set and save
the model in the directory specified by `--save_dir`.


`cifar10_simclr.py`
This script's main puporse is to evaluate the SimCLR method with different combination of data augmentation strategies.

By default, the data augmentation combination is `random crop, resize, random horinzontal flip, color distortion, gaussian blur`.
Otherwise it is possible to specify a combination of two transforms. For example:

```
python3 cifar10_simclr.py --data_dir data --first_transform crop --second_transform color
```

Just like the script `cifar10_autoencoder` this one trains (unsupervised) the SimCLR model with ResNet18 encoder on cifar10 train set and save
the model in the directory specified by `--save_dir`.


`linear_classification.py`
The linear evaluation method of a pretrained model. To run this script, a pretained model saved on disk is need.

Examples:

```
python3 linear_classification.py models/simclr_crop_color_40.pt --model_name simclr
python3 linear_classification.py models/resnet_ae_100.pt --model_name resnet_ae
python3 linear_classification.py models/dae_0.001_32_100.pt --model_name vanilla_dae
```

Just like `cifar10_supervised.py`, this script will evaluate the given model on 1%, 10%, 50% and 100% of the train features and ouputs the best test accuarcies
at the end of each phase.

`models.py`: Contain the autoencoder with three conv layers as encoder and autoencoder with resnet18 as encoder.

`utils.py`: Some utility functions shared between the scrips such as `inference`, `class_balanced_subset`.


## RAW Results (that were transformed to table and heatmaps in report)

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
lr 1e-3

bs 32

100 epochs

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
lr 1e-3

bs 32

100 epochs

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

#### Unsupervised pretraining on CIFAR-10 train set hyperparams
epochs: 40

learning rate: 3e-4

batch size: 256

projection dim: 46

temperature of contrastive loss: 0.5


#### Best Test Accuracy of linear classifier with different combinations of transforms
epochs: 100

learning rate: 1e-3

batch size: 32

**crop + color + blur** (Optimal Combination of transforms according to paper)

1%  48.89

10%  55.24

50%  57.1700

100%  58.51


**crop + crop**

1% 34.18

10% 41.44

50% 44.45

100% 45.87

**crop + color**

1% 48.79

10% 54.88

50% 57.82

100% 58.54

**crop + blur**

1% 34.91

10% 42.87

50% 46.41

100% 47.04


**color + crop**

1% 48.17

10% 54.67

50% 57.70

100% 58.16

**color + color**

1% 27.03

10% 34.13

50% 36.55

100% 37.10

**color + blur**

1% 25.56

10% 32.73

50% 35.21

100% 36.02


**blur + crop**

1% 35.00

10% 42.99

50% 46.46

100% 47.41

**blur + color**

1% 25.52

10% 32.52

50% 34.23

100% 35.53

**blur + blur**

1% 26.20

10% 34.01

50% 35.82

100% 36.91%

