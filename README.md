## ConvLSR-Net
This is the code for our paper:

* [R. Zhang, Q. Zhang and G. Zhang, "LSRFormer: Efficient Transformer Supply Convolutional Neural Networks with Global Information for Aerial Image Segmentation" in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2024.3366709.](https://ieeexplore.ieee.org/document/10438484)

  
## Results

We repeated the experiment with 5 different random seeds. The average and best results of the 5 repetitions are as follows:

|    Method     |  Dataset  |  mIoU (Average) | mIoU (Best) |
|:-------------:|:---------:|:-----:|:-----:|------:|
|  ConvLSR-Net   |   iSAID   |  70.8±0.11  | 70.89 |
|  ConvLSR-Net   | Vaihingen |  84.56±0.06 | 84.64 |
|  ConvLSR-Net   |  Potsdam  |  87.80±0.08 | 87.91 |
|  ConvLSR-Net   |  LoveDA   |  54.77±0.08 | [54.86](https://codalab.lisn.upsaclay.fr/my/competition/submission/340641/detailed_results)|

Due to some random operations in the training stage, reproduced results (run once) may slightly different from the reported in paper.


## Data Preprocessing

Please follw the [GeoSeg](https://github.com/WangLibo1995/GeoSeg) to prepocess the LoveDA, Potsdam and Vaihingen dataset.

Please follow the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isaid) to prepocess the iSAID dataset. 



## Training

"-c" means the path of the config, use different **config** to train different models.

```shell
python train_supervision.py -c ./config/isaid/convlsrnet.py
```

```shell
python train_supervision_dp.py -c ./config/potsdam/convlsrnet.py
```

```shell
python train_supervision_dp.py -c ./config/vahingen/convlsrnet.py
```

```shell
python train_supervision_dp.py -c ./config/loveda/convlsrnet.py
```

## Testing

**iSAID** 
```shell
python test_isaid.py -c ./config/isaid/convlsrnet.py -o ~/fig_results/isaid/convlsrnet_isaid/  -t "d4"
```

**Vaihingen**
```shell
python test_vaihingen.py -c ./config/vaihingen/convlsrnet.py -o ~/fig_results/convlsrnet_vaihingen/ --rgb -t "d4"
```

**Potsdam**
```shell
python test_potsdam.py -c ./config/potsdam/convlsrnet.py -o ~/fig_results/convlsrnet_potsdam/ --rgb -t "d4"
```

**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))

My LoveDA results: [LoveDA Test Results](https://codalab.lisn.upsaclay.fr/my/competition/submission/340641/detailed_results/)

```shell
python test_loveda.py -c ./config/loveda/convlsrnet.py -o ~/fig_results/convlsrnet_loveda --rgb -t "d4"
```


## Citation and Contact

If you find this project useful in your research, please consider citing our papers：

* R. Zhang, Q. Zhang and G. Zhang, "LSRFormer: Efficient Transformer Supply Convolutional Neural Networks with Global Information for Aerial Image Segmentation," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2024.3366709.

```shell
@ARTICLE{10438484,
  author={Zhang, Renhe and Zhang, Qian and Zhang, Guixu},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={LSRFormer: Efficient Transformer Supply Convolutional Neural Networks with Global Information for Aerial Image Segmentation}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2024.3366709}}
```

If you encounter any problems while running the code, feel free to contact me via [stdcoutzrh@gmail.com](stdcoutzrh@gmail.com). Thank you!


## Acknowledgement

Our training scripts comes from [GeoSeg](https://github.com/WangLibo1995/GeoSeg). Thanks for the author's open-sourcing code.
- [GeoSeg(UNetFormer)](https://github.com/WangLibo1995/GeoSeg)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
