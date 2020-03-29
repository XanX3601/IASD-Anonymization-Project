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
