# [Leveraging Unlabeled Data for 3D Medical Image Segmentation through Self-Supervised Contrastive Learning](xmindflow.com)
Welcome to our GitHub repository! Our 3D semi-supervised segmentation approach addresses key challenges by leveraging two specialized subnetworks, correcting errors and enhancing contextual information. We introduce targeted verification training and self-supervised contrastive learning to improve predictions. Our model demonstrates superior performance on clinical MRI and CT scans for organ segmentation, outperforming state-of-the-art methods. Dive into our code for advanced 3D segmentation capabilities!

#### Please consider starring us, if you found it useful. Thanks

## Updates
- November 21, 2023: First release of the code.

## Quick Overview
![Diagram of the proposed method]()


## Installation
This code has been implemented in python language using Pytorch libarary and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:
* CentOS Linux release 7.3.1611
* Python 3.6.13
* CUDA 9.2
* PyTorch 1.9.0
* medpy 0.4.0
* tqdm，h5py

## Getting Started
Please download the prepared dataset from the following link and use the dataset path in the training and evalution code.
* [LA Dataset](https://drive.google.com/drive/folders/1_LObmdkxeERWZrAzXDOhOJ0ikNEm0l_l)
* [Pancreas Dataset](https://drive.google.com/drive/folders/1kQX8z34kF62ZF_1-DqFpIosB4zDThvPz)

Please change the database path and data partition file in the corresponding code.

### Training
To train the network on the LA dataset, execute python `pyhon train_LA`. For the Pancreas dataset, use python `pyhon train_pancreas`
### Evaluation
To evaluate the network on the LA dataset, run `pyhon test_LA`. For the Pancreas dataset, run `pyhon test_pancreas`



## Citation
If you find this project useful, please consider citing:

```bibtex
@InProceedings{MCF,
    author    = {Wang, Yongchao and Xiao, Bin and Bi, Xiuli and Li, Weisheng and Gao, Xinbo},
    title     = {MCF: Mutual Correction Framework for Semi-Supervised Medical Image Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {15651-15660}
}
```
## Acknowledgement

We build the project based on MCF-semsupervise.
Thanks for their contribution.


