import torch.nn as nn

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
