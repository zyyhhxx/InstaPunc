from torch.utils.data import Dataset, DataLoader

class PuncDataset(Dataset):
    def __init__(self, data, labels):
        self.data_range = data[0]
        self.labels = labels
        self.data = data[1]
        
        if len(labels[0]) != len(self.data_range) or len(labels[1]) != len(self.data_range):
            print("Dataset sizes mismatch!")

    def __len__(self):
        return len(self.data_range)

    def __getitem__(self, index):
        data_index, data_left, data_right = self.data_range[index]
        return self.data[data_index][data_left:data_right], (self.labels[0][index], self.labels[1][index])

class PuncInferenceDataset(Dataset):
    def __init__(self, data):
        self.data_range = data[0]
        self.data = data[1]

    def __len__(self):
        return len(self.data_range)

    def __getitem__(self, index):
        data_index, data_left, data_right = self.data_range[index]
        return self.data[data_index][data_left:data_right]