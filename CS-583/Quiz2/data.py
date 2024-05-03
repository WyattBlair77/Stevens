import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from torch import FloatTensor


train_test_split = 0.8
sequence_length = 16
shuffle = True
batch_size = 1

# Need to make some custom extensions of the PyTorch Dataset and DataLoader classes to handle dates etc.
class CustomDataset(Dataset):

    def __init__(self, dataset, sequence_length):
        
        super().__init__()

        self.sequence_length = sequence_length
        self.dataset = dataset.reset_index()
        self.x_sequences, self.y_sequences = self.create_sequences()

        self.x = TensorDataset(self.x_sequences)
        self.y = TensorDataset(self.y_sequences[:, -1])

    def create_sequences(self):

        x_sequences = []
        y_sequences = []

        for i in range(len(self.dataset) - self.sequence_length):
            
            x_seq = FloatTensor(self.dataset['DATE-INDEX'][i : i+self.sequence_length].values)
            x_sequences.append(x_seq)

            y_seq = FloatTensor(self.dataset['POINTS'][i : i+self.sequence_length].values)
            y_sequences.append(y_seq)

        return torch.stack(x_sequences), torch.stack(y_sequences)  # Stack for a new dimension

    def __getitem__(self, index):

        x = self.x[index]
        y = self.y[index]

        return x, y
    
    def __len__(self):

        assert len(self.x) == len(self.y)
        return len(self.x)

class CustomDataLoader(DataLoader):

    def __init__(self, dataset, batch_size, shuffle=False, sequence_length=sequence_length):

        dataset = CustomDataset(dataset, sequence_length=sequence_length)
        self.x = dataset.x
        self.y = dataset.y

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def __getitem__(self, index):

        return self.dataset[index]

def perform_fft(data):

    fft_data = data.copy()

    y_index_space = torch.Tensor(fft_data['POINTS'])
    y_freq_space = torch.fft.fft(y_index_space)
    fft_data['POINTS'] = y_freq_space
    
    return fft_data

def get_data():
    
    try:
        data = pd.read_csv('./data/Electric_Production.csv', index_col='DATE', date_format='%m/%d/%y')
    except:
        data = pd.read_csv('../data/Electric_Production.csv', index_col='DATE', date_format='%m/%d/%y')

    data.rename(columns={'IPG2211A2N': 'POINTS'}, inplace=True)

    latest_train_ind = int(len(data.index) * train_test_split)
    latest_train_date = list(data.index)[latest_train_ind]

    # scale data
    scaler = MinMaxScaler()
    norm_data = pd.DataFrame(data=scaler.fit_transform(X=data), columns=data.columns, index=data.index)

    # add date index for ease of use later
    norm_data['DATE-INDEX'] = np.arange(len(data.index))

    # put the data in frequency space using fft
    fft_data = perform_fft(norm_data)

    # split df
    train_df = fft_data.loc[:latest_train_date]
    test_df = fft_data.drop(index=train_df.index)

    # create torch CustomDataLoader
    dataloader = CustomDataLoader(fft_data, batch_size=1, shuffle=False, sequence_length=sequence_length)

    # split dataloader for train and test
    split_index = int(len(dataloader) * train_test_split)
    train_subset = Subset(dataloader, range(0          , split_index    ))
    test_subset  = Subset(dataloader, range(split_index, len(dataloader)))

    # create dataloaders
    train_loader_shuffle = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_subset, batch_size=1)

    out = {
        'data': data,
        'norm_data': norm_data,
        'fft_data': fft_data,

        'train_df': train_df,
        'test_df': test_df,

        'latest_train_date': latest_train_date,
        'split_index': split_index,
        'sequence_length': sequence_length,
        'batch_size': batch_size,

        'dataloader': dataloader,
        'train_subset': train_subset,
        'train_loader': train_loader,
        'train_loader_shuffle': train_loader_shuffle,
        'test_subset': test_subset,
        'test_loader': test_loader,

    }

    return out
