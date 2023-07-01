## LSRFormer: Efficient Transformer Supply Convolutional Neural Networks with Global Information for Aerial Image Segmentation

This is the official Pytorch/PytorchLightning implementation of the paper:

* LSRFormer: Efficient Transformer Supply Convolutional Neural Networks with Global Information for Aerial Image Segmentation.

## Results
|    Method     |  Dataset  |  mF1   |  OA   |  mIoU |
|:-------------:|:---------:|:-----:|:-----:|------:|
|  ConvLSR-Net   |   iSAID   |   -   |   -   | 70.6 |
|  ConvLSR-Net   | Vaihingen | 91.4 | 91.8 | 84.4 |
|  ConvLSR-Net  |  Potsdam  | 93.5 | 92.0 | 87.9 |
|  ConvLSR-Net   |  LoveDA   |   -   |   -   | 54.9 |

Due to some random operations in the training stage, reproduced results (run once) are slightly different from the reported in paper.


## Introduction

LSRFormer: Efficient Transformer Supply Convolutional Neural Networks with Global Information for Aerial Image Segmentation


## Pretrained Weights

We will upload trained weighs as soon as possible.

## Data Preprocessing

Please follw the [GeoSeg](https://github.com/WangLibo1995/GeoSeg) to prepocess the LoveDA, Potsdam and Vaihingen dataset.

Please follow the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isaid) to prepocess the iSAID dataset.



## Training

"-c" means the path of the config, use different **config** to train different models.

```shell
python train_supervision.py -c ./config/isaid/lsrformer.py
```

```shell
python train_supervision_dp.py -c ./config/potsdam/lsrformer.py
```

```shell
python train_supervision_dp.py -c ./config/vahingen/lsrformer.py
```

```shell
python train_supervision_dp.py -c ./config/loveda/lsrformer.py
```

## Testing

**iSAID** 
```shell
python test_isaid.py -c ./config/isaid/lsrformer.py -o ~/fig_results/isaid/lsrformer_isaid/  -t "d4"
```

**Vaihingen**
```shell
python test_vaihingen.py -c ./config/vaihingen/lsrformer.py -o ~/fig_results/lsrformer_vaihingen/ --rgb -t "d4"
```

**Potsdam**
```shell
python test_potsdam.py -c ./config/potsdam/lsrformer.py -o ~/fig_results/lsrformer_potsdam/ --rgb -t "d4"
```

**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))

my results: [LoveDA Test Results](https://codalab.lisn.upsaclay.fr/my/competition/submission/340641/detailed_results/)

```shell
python test_loveda.py -c ./config/loveda/lsrformer.py -o ~/fig_results/lsrformer_loveda --rgb -t "d4"
```


## Acknowledgement

Our training scripts comes from GeoSeg. Thanks for the author's open source code.
- [GeoSeg(UNetFormer)](https://github.com/WangLibo1995/GeoSeg)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
