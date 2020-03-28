import pickle
import numpy as np

cifar100_directory = 'cifar-100-python/'

def load_cifar100():
    load = lambda file: pickle.load(file, encoding='bytes')

    with open('{}train'.format(cifar100_directory), 'rb') as train_file:
        train_dict = load(train_file)

    with open('{}test'.format(cifar100_directory), 'rb') as test_file:
        test_dict = load(test_file)

    with open('{}meta'.format(cifar100_directory), 'rb') as meta_file:
        meta_dict = load(meta_file)


    x_train = np.reshape(np.array(train_dict[b'data']), (50000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float32) / 255
    x_test = np.reshape(np.array(test_dict[b'data']), (10000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float32) / 255
    x = np.concatenate((x_train, x_test), axis=0)

    coarse_labels_train = np.array(train_dict[b'coarse_labels']).astype(np.uint8)
    coarse_labels_test = np.array(test_dict[b'coarse_labels']).astype(np.uint8)
    coarse_labels = np.concatenate((coarse_labels_train, coarse_labels_test), axis=0)

    labels_train = np.array(train_dict[b'fine_labels']).astype(np.uint8)
    labels_test = np.array(test_dict[b'fine_labels']).astype(np.uint8)
    labels = np.concatenate((labels_train, labels_test), axis=0)

    fine_label_names = np.array(meta_dict[b'fine_label_names'], dtype=np.str)
    coarse_label_names = np.array(meta_dict[b'coarse_label_names'], dtype=np.str)

    return x, coarse_labels, labels, fine_label_names, coarse_label_names
