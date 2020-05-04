import torch.nn as nn

# The baseline model
class PuncLinear(nn.Module):
    def __init__(self, classes, window_size):
        super(PuncLinear, self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(300 * window_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(classes)),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, data):
        flatten_data = data.reshape(data.shape[0], 1500)
        prediction = self.linear(flatten_data)
        
        return prediction
