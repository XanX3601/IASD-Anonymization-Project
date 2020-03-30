import imgaug.augmenters as iaa
import numpy as np
import torch.utils.data as data


class Dataset(data.dataset.Dataset):
    def __init__(self, x_train, y_train, label=0):
        self.x = x_train
        self.y = y_train
        self.label = label

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def augment_data(x_train, y_train):
    x_train = x_train.transpose(0, 2, 3, 1)
    augments = []

    x_train_aug_0 = iaa.Flipud(1.0)(images=x_train)
    augments.append(x_train_aug_0)

    x_train_aug_1 = iaa.Fliplr(1.0)(images=x_train)
    augments.append(x_train_aug_1)

    seq = iaa.Sequential([iaa.GaussianBlur(sigma=1.5)])
    x_train_aug_2 = seq(images=x_train)
    augments.append(x_train_aug_2)

    seq = iaa.Sequential([iaa.Affine(scale={"x": (1.2, 1.5), "y": (1.2, 1.5)})])
    x_train_aug_3 = seq(images=x_train)
    augments.append(x_train_aug_3)

    for aug in augments:
        x_train = np.concatenate((x_train, aug), 0)
        y_train = np.concatenate((y_train, y_train), 0)

    x_train = x_train.transpose(0, 3, 1, 2)
    return x_train, y_train
