# IASD-Anonymization-Project
This project aims to reproduce the results of this [article](https://arxiv.org/abs/1306.4447) with artificial neural networks on the CIFAR-100 dataset.

## The paper
The paper "Hacking Smart Machines with Smarter Ones: How to Extract Meaningful Data from Machine Learning Classifiers" has been written in 2013. You can find the original article and a detailed review of it in the [`materials`](./materials) folder.

## Prerequisites
You need to have Python 3.7 (at least) installed on your system. To start, please run the following commands.

```shell
git clone https://github.com/XanX3601/IASD-Anonymization-Project.git
pip install -r requirements.txt
```

## How to use
TODO: complete this part with the associated commands.
* Create a neural network able to recognize if there is a vehicle or not in a picture.
* Train it on CIFAR-100 pictures. The neural network output is binary.
* Train several datasets in order to obtain several neural networks models.
* Train a meta-classifier on this different neural networks.
* Use it to infer some information from the first neural network.
