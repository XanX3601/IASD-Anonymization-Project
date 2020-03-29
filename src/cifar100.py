from .dataset import Dataset
import pickle
import numpy as np

cifar100_directory = 'cifar-100-python/'

datasets = []

def load_cifar100():
    load = lambda file: pickle.load(file, encoding='bytes')

    with open('{}train'.format(cifar100_directory), 'rb') as train_file:
        train_dict = load(train_file)

    with open('{}test'.format(cifar100_directory), 'rb') as test_file:
        test_dict = load(test_file)

    with open('{}meta'.format(cifar100_directory), 'rb') as meta_file:
        meta_dict = load(meta_file)


    x_train = np.reshape(np.array(train_dict[b'data']), (50000, 3, 32, 32)).astype(np.float32) / 255
    x_test = np.reshape(np.array(test_dict[b'data']), (10000, 3, 32, 32)).astype(np.float32) / 255
    x = np.concatenate((x_train, x_test), axis=0)

    coarse_labels_train = np.array(train_dict[b'coarse_labels']).astype(np.float32)
    coarse_labels_test = np.array(test_dict[b'coarse_labels']).astype(np.float32)
    coarse_labels = np.concatenate((coarse_labels_train, coarse_labels_test), axis=0)

    labels_train = np.array(train_dict[b'fine_labels']).astype(np.float32)
    labels_test = np.array(test_dict[b'fine_labels']).astype(np.float32)
    labels = np.concatenate((labels_train, labels_test), axis=0)

    fine_label_names = np.array(meta_dict[b'fine_label_names'], dtype=np.str)
    coarse_label_names = np.array(meta_dict[b'coarse_label_names'], dtype=np.str)

    return x, coarse_labels, labels, fine_label_names, coarse_label_names

def populate_datasets():
    np.random.seed(42)

    images, labels, sub_labels, sub_label_names, label_names = load_cifar100()

    bicycle_index = np.where(sub_label_names == 'bicycle')[0][0]
    filter_bicycle = sub_labels == bicycle_index

    vehicle_indexes = np.where((label_names == 'vehicles_1') | (label_names == 'vehicles_2'))[0]
    filter_vehicles = np.isin(labels, vehicle_indexes)

    filter_vehicles_non_bicyle = filter_vehicles & np.logical_not(filter_bicycle)

    filter_non_vehicles = np.logical_not(filter_vehicles)

    images_bicyle = images[filter_bicycle]
    images_vehicles_non_bicycle = images[filter_vehicles_non_bicyle]
    images_non_vehicles = images[filter_non_vehicles]

    np.random.shuffle(images_bicyle)
    np.random.shuffle(images_vehicles_non_bicycle)
    np.random.shuffle(images_non_vehicles)

    x = np.concatenate((images_vehicles_non_bicycle[:540], images_non_vehicles[:540]))
    y = np.concatenate((np.full((540, 1), 1.0), np.full((540, 1), 0.0)))
    indexes = np.arange(x.shape[0])
    np.random.shuffle(indexes)
    x = x[indexes]
    y = y[indexes].astype(np.float32)

    dataset_original = Dataset(x, y)

    datasets.append(dataset_original)

    images_vehicles_non_bicycle = images_vehicles_non_bicycle[540:]
    images_non_vehicles = images_non_vehicles[540:]

    nb_bicycle_per_dataset = images_bicyle.shape[0] // 5
    nb_vehicle_non_bicycle_per_dataset = images_vehicles_non_bicycle.shape[0] // 2 // 5

    for i in range(5):
        x = np.concatenate(
            (
                images_bicyle[i * nb_bicycle_per_dataset : (i + 1) * nb_bicycle_per_dataset],
                images_vehicles_non_bicycle[i * nb_vehicle_non_bicycle_per_dataset: (i + 1) * nb_vehicle_non_bicycle_per_dataset],
                images_non_vehicles[np.random.choice(images_non_vehicles.shape[0], nb_vehicle_non_bicycle_per_dataset + nb_bicycle_per_dataset, replace=False)]
            )
        )
        y = np.concatenate(
            (
                np.full((nb_bicycle_per_dataset + nb_vehicle_non_bicycle_per_dataset, 1), 1.0),
                np.full((nb_bicycle_per_dataset + nb_vehicle_non_bicycle_per_dataset, 1), 0.0)
            )
        )
        indexes = np.arange(x.shape[0])
        np.random.shuffle(indexes)
        x = x[indexes]
        y = y[indexes].astype(np.float32)

        datasets.append(Dataset(x, y, 1))

    images_vehicles_non_bicycle = images_vehicles_non_bicycle[nb_vehicle_non_bicycle_per_dataset * 5:]

    for i in range(5):
        x = np.concatenate(
            (
                images_vehicles_non_bicycle[i * nb_vehicle_non_bicycle_per_dataset: (i + 1) * nb_vehicle_non_bicycle_per_dataset],
                images_non_vehicles[np.random.choice(images_non_vehicles.shape[0], nb_vehicle_non_bicycle_per_dataset, replace=False)]
            )
        )
        y = np.concatenate(
            (
                np.full((nb_vehicle_non_bicycle_per_dataset, 1), 1.0),
                np.full((nb_vehicle_non_bicycle_per_dataset, 1), 0.0)
            )
        )
        indexes = np.arange(x.shape[0])
        np.random.shuffle(indexes)
        x = x[indexes]
        y = y[indexes].astype(np.float32)

        datasets.append(Dataset(x, y, 0))
