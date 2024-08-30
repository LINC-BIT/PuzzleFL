# PuzzleFL

## Table of contents
- [1 Introduction](#1-introduction)
- [2 How to get started](#2-how-to-get-started)
  * [2.1 Setup](#21-setup)
  * [2.2 Usage](#22-usage)
- [3 Supported models](#3-supported-models-in-image-classification)
- [4 Experiments setting](#4-Experiments-setting)
  * [4.1 Generate task](#41-Generate-task)
  * [4.2 Selection of hyperparameters](#42-Selection-of-hyperparameters)
- [5 Experiments](#5-Experiments)
  * [5.1 Under different workloads (model and dataset)](#51-under-different-workloads-model-and-dataset)
  * [5.2 Under different network bandwidths](#52-under-different-network-bandwidths)
  * [5.3 Large scale](#53-large-scale)
  * [5.4 Long task sequence](#54-long-task-sequence)
  * [5.5 Under different parameter settings](#55-under-different-parameter-settings)
  * [5.6 Applicability on different networks](#56-applicability-on-different-networks)

## 1 Introduction
PuzzleFL is designed to achieve SOTA performance (accuracy, time, and communication cost etc.) in decetralized federated continual learning setting. It currently supports six differnet networks of image/text classification: ResNet, MobiNet, DenseNet, ViT, RNN, LSTM and Bert. 
- [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html): this model consists of multiple convolutional layers and pooling layers that extract the information in image. Typically, ResNet suffers from gradient vanishing (exploding) and performance degrading when the network is  deep. ResNet thus adds BatchNorm to alleviate gradient vanishing (exploding) and adds residual connection to alleviate the performance degrading.
- [MobileNet](https://arxiv.org/abs/1801.04381): MobileNet is a lightweight convolutional network which widely uses the depthwise separable convolution.
- [DenseNet](https://arxiv.org/pdf/1707.06990.pdf): DenseNet extends ResNet by adding connections between each blocks to aggregate all multi-scale features.
- [Vit](): The Vision Transformer (ViT) applies the Transformer architecture to image recognition tasks. It segments the image into multiple patches, then inputs these small blocks as sequence data into the Transformer model, using the self-attention mechanism to capture global and local information within the image, thereby achieving efficient image classification.
- [RNN](): RNN (Recurrent Neural Network) is a type of neural network specifically designed for sequential data, excelling at handling time series and natural language with temporal dependencies.
- [LSTM](): LSTM (Long Short-Term Memory) is a special type of RNN that can learn long-term dependencies, suitable for tasks like time series analysis and language modeling.
- [Bert](): BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language representation model based on the Transformer architecture, which captures contextual information in text through deep bidirectional training. The BERT model excels in natural language processing (NLP) tasks and can be used for various applications such as text classification, question answering systems, and named entity recognition.

## 2 How to get started
### 2.1 Setup
**Requirements**
- Edge devices such as Jetson AGX, Jetson TX2, Jetson Xavier NX, Jetson Nano and Rasperry Pi.
- Linux and Windows 
- Python 3.6+
- PyTorch 1.9+
- CUDA 10.2+ 

**Preparing the virtual environment**

1. Create a conda environment and activate it.
	```shell
	conda create -n FedKNOW python=3.7
	conda active FedKNOW
	```
	
2. Install PyTorch 1.9+ in the [offical website](https://pytorch.org/). A NVIDIA graphics card and PyTorch with CUDA are recommended.

  ![image](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ec360791671f4a4ab322eb4e71cc9e62~tplv-k3u1fbpfcp-zoom-1.image)

3. Clone this repository and install the dependencies.
  ```shell
  git clone https://github.com/LINC-BIT/PuzzleFL/
  pip install -r requirements.txt
  ```
### 2.2 Usage
Run PuzzleFL or the baselines:
```shell
python ClientTrainPuzzleFL/main_PuzzleFL.py(or other baselines) --dataset [dataset] --model [mdoel]
--num_users [num_users]  --shard_per_user [shard_per_user] --frac [frac] 
--local_bs [local_bs] --lr [lr] --task [task] --epoch [epoch]  --local_ep 
[local_ep] --local_local_ep [local_local_ep] --store_rate [store_rate] 
--select_grad_num [select_grad_num] --gpu [gpu]
```
Arguments:

- `dataset` : the dataset, e.g. `cifar100`, `MiniImageNet`, `TinyImageNet`, `ASC`, `DSC`

- `model`: the model, e.g. `6-Layers CNN`, `ResNet18`, `DenseNet`, `MobiNet`, `RNN`, `LSTM`, `Bert`

- `num_users`: the number of clients

- `shard_per_user`: the number of classes in each client

- `neighbor_nums`: the number of clients per neighbor

- `local_bs`: the batch size in each client

- `lr`: the learning rate

- `task`: the number of tasks

- `epochs`: the number of communications between each client

- `local_ep`:the number of epochs in clients

- `local_local_ep`:the number of updating the local parameters in clients

- `store_rate`: the store rate of model parameters in FedKNOW

- `select_grad_num`: the number of choosing the old grad in FedKNOW

- `gpu`: GPU id

  More details refer to `utils/option.py`.

## 3 Supported models in image/text classification
||Model Name|Data|Script|
|--|--|--|--|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[6 layer_CNN (NeurIPS'2020)](https://proceedings.neurips.cc/paper/2020/hash/258be18e31c8188555c2ff05b4d542c3-Abstract.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;<br>&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://proceedings.neurips.cc/paper/2020/hash/258be18e31c8188555c2ff05b4d542c3-Abstract.html) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;| &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/ResNet.sh) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[ResNet (CVPR'2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;<br>&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;| &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/ResNet.sh) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[MobileNetV2 (CVPR'2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/MobileNet.sh) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[DenseNet(CVPR'2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/DenseNet.sh)&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[ViT(ICLR'2021)](https://iclr.cc/virtual/2021/oral/3458) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/DenseNet.sh)&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[RNN (CVPR'2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[ASC](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/SENet.sh)&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[LSTM (CVPR'2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[ASC](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/SENet.sh)&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[Bert (CVPR'2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[ASC](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/SENet.sh)&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;|
