from torch.utils.data import Dataset, DataLoader

class PuncDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data
        
        if len(labels) != len(data):
            print("Dataset sizes mismatch!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Select sample
        return self.data[index], self.labels[index]