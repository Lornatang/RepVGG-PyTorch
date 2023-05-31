# RepVGG-PyTorch

PyTorch implements `RepVGG: Making VGG-style ConvNets Great Again` paper.

# ResNet-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697v3.pdf).

## Table of contents

- [RepVGG-PyTorch](#resnet-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test RepVGG-A0](#test-repvgg-a0)
        - [Train RepVGG](#train-repvgg-a0)
        - [Resume train RepVGG](#resume-train-repvgg-a0)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [RepVGG: Making VGG-style ConvNets Great Again](#repvgg-making-vgg-style-convnets-great-again)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify yaml file.

### Test RepVGG-A0

### Train RepVGG-A0

### Resume train RepVGG-A0

## Result

Source of original paper results: [https://arxiv.org/pdf/1512.03385v1.pdf](https://arxiv.org/pdf/1512.03385v1.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|   Model   |   Dataset   | Top-1 error (val)  | Top-5 error (val) |
|:---------:|:-----------:|:------------------:|:-----------------:|
| resnet18  | ImageNet_1K | 27.88%(**30.25%**) |   -(**10.93%**)   |
| resnet34  | ImageNet_1K | 25.03%(**26.71%**) | 7.76%(**8.58%**)  |
| resnet50  | ImageNet_1K | 22.85%(**19.65%**) | 6.71%(**4.87%**)  |
| resnet101 | ImageNet_1K | 21.75%(**18.33%**) | 6.05%(**4.34%**)  |
| resnet152 | ImageNet_1K | 21.43%(**17.66%**) | 5.71%(**4.08%**)  |

```bash
# Download `ResNet18-ImageNet_1K-57bb63e.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `resnet18` model successfully.
Load `resnet18` model weights `/ResNet-PyTorch/results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar` successfully.
tench, Tinca tinca                                                          (91.46%)
barracouta, snoek                                                           (7.15%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (0.43%)
coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch    (0.27%)
platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus (0.21%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### RepVGG: Making VGG-style ConvNets Great Again

*Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, Jian Sun*

##### Abstract

We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a
stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time
architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80% top-1
accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50
or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the state-of-the-art models like
EfficientNet and RegNet. The code and trained models are available at this https URL.

[[Paper]](https://arxiv.org/pdf/2101.03697v3.pdf) [[Code]](https://github.com/megvii-model/RepVGG.)

```bibtex
@misc{ding2021repvgg,
      title={RepVGG: Making VGG-style ConvNets Great Again}, 
      author={Xiaohan Ding and Xiangyu Zhang and Ningning Ma and Jungong Han and Guiguang Ding and Jian Sun},
      year={2021},
      eprint={2101.03697},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```