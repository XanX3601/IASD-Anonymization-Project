# IASD-Anonymization-Project
This project aims to reproduce the results of this [article](https://arxiv.org/abs/1306.4447) with artificial neural networks on the CIFAR-100 dataset.

## The paper
The paper "Hacking Smart Machines with Smarter Ones: How to Extract Meaningful Data from Machine Learning Classifiers" has been written in 2013. You can find the original article and a detailed review of it in the [`materials`](./materials) folder. A report can also be found for a more detailed explanation of the project.

## Prerequisites
You need to have Python 3.7 (at least) installed on your system. To start, please run the following commands.

```shell
git clone https://github.com/XanX3601/IASD-Anonymization-Project.git
pip install -r requirements.txt
```

## Goal
We train a target classifier to recognize if a image contains a vehicle or not. The dataset does not include any bicycles. Our goal, is to create a meta-classifier which will be able to tell if a classifier has been trained on a dataset containing bicyles images or not. Labels meta-classifier datasets are: 0 if there are no bicycles and 1 otherwise.

## How to use
You have 2 scripts, `script.py` doest 1 pass, and `script_stats.py` does multiples passes in order to collect statistics. Otherwise, you can use the following commands yourself.

* Download CIFAR-100 data.
```shell
python download_data.py
```
* Create necessary directories.
```shell
mkdir networks networks/classifiers meta_datasets
```
* Create the target classifier. Add `--cuda` if you want to use a GPU.
```shell
python create_net.py --path networks/target_classifier.pt
```
* Create the other classifiers. Add `--cuda` if you want to use a GPU.
```shell
for i in {1..10}; do python create_net.py --path "networks/classifiers/classifier_$i.pt"; done
```
* Train the target classifier on a dataset without bicycles. Add `--cuda` if you want to use a GPU.
```shell
python train_net.py --path networks/target_classifier.pt --dataset 0
```
* Train the other classifiers. Add `--cuda` if you want to use a GPU.
```shell
for i in {1..10}; do echo "Training classifier $i"; python train_net.py --path "networks/classifiers/classifier_$i.pt" --dataset "$i"; done
```
* Extract weights of all classifiers except the target one. Add `--cuda` if you want to use a GPU.
```shell
python extract_weights.py --dir networks/classifiers --out meta_datasets
```
* Create the meta-classifier. Add `--cuda` if you want to use a GPU. Note that if you change the classifier architecture, you need to adjuste the input size vector.
```shell
python create_meta_net.py --path networks/meta_classifier.pt --input-size-vector 3200
```
* Train the meta-classifier. Add `--cuda` if you want to use a GPU.
```shell
python train_meta_net.py --path networks/meta_classifier.pt --dataset meta_datasets
```
* Use the meta-classifier to finally check if the target classifier was trained on a dataset containing bikes. Add `--cuda` if you want to use a GPU.
```shell
python bikes_or_not_bikes_that_is_the_question.py --target networks/target_classifier.pt --meta networks/meta_classifier.pt
```
