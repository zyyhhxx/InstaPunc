import torch.nn as nn
import torch.nn.functional as F

# The baseline model
class PuncLinear(nn.Module):
    def __init__(self, classes, window_size):
        super(PuncLinear, self).__init__()
        
        self.linear_punc = nn.Sequential(
            nn.Linear(300 * window_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(classes)),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.linear_cap = nn.Sequential(
            nn.Linear(300 * window_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, data):
        flatten_data = data.reshape(data.shape[0], 1500)
        prediction_punc = self.linear_punc(flatten_data)
        prediction_cap = self.linear_cap(flatten_data)
        
        return prediction_punc, prediction_cap

# The lstm model
class PuncLstm(nn.Module):
    def __init__(self, classes, window_size):
        super(PuncLstm, self).__init__()
        
        input_size = 300
        hidden_size = 150
        self.sequence_size = 300 * window_size
        self.hidden = None

        self.lstm = nn.LSTM(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
            )

        self.linear_punc = nn.Sequential(
            nn.Linear(self.sequence_size, self.sequence_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.sequence_size // 2, self.sequence_size // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.sequence_size // 4, len(classes)),
            nn.ReLU()
        )

        self.linear_cap = nn.Sequential(
            nn.Linear(self.sequence_size, self.sequence_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.sequence_size // 2, self.sequence_size // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.sequence_size // 4, 2),
            nn.ReLU()
        )

    def forward(self, data):
        if self.hidden is None:
            out, self.hidden = self.lstm(data)
        else:
            self.hidden = tuple([e.data for e in self.hidden])
            out, self.hidden = self.lstm(data, self.hidden)
        flatten_data = data.reshape(out.shape[0], self.sequence_size)
        prediction_punc = self.linear_punc(flatten_data)
        prediction_cap = self.linear_cap(flatten_data)
        
        return prediction_punc, prediction_cap