import pandas as pd
from tqdm.auto import tqdm
from .constants import CLASSES
import torch

def train(train_loader, dev_loader, model, criterion, optimizer, epochs, device):

    for epoch in range(epochs):  # loop over the dataset multiple times
        print("----------------------------")
        print("Epoch:", epoch)
        
        model.train()
        t = tqdm(iter(train_loader), leave=False, total=len(train_loader))
        epoch_loss = 0.0
        
        for _, data in enumerate(t, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += outputs.shape[0] * loss.item()
            
        print("Training loss:", epoch_loss / len(train_loader))
        _, acc = validate(dev_loader, model, device)
        print('Validation accuracy: %d %%' % (100 * acc))
                
def validate(dataloader, model, device):
    correct = 0
    total = 0
    predicted_total = [1] * len(CLASSES)
    predicted_correct = [0] * len(CLASSES)
    predicted_expected = [1] * len(CLASSES)
    
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Gathering information for f score
            for i in range(predicted.shape[0]):
                predicted_class = predicted[i]
                correct_class = labels[i]
                predicted_total[predicted_class] += 1
                predicted_expected[correct_class] += 1
                if predicted_class == correct_class:
                    predicted_correct[predicted_class] += 1
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    f_scores = []
    for i in range(len(CLASSES)):
        precision = predicted_correct[i] / predicted_total[i]
        recall = predicted_correct[i] / predicted_expected[i]
        f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        f_scores.append([CLASSES[i], predicted_total[i], predicted_correct[i], predicted_expected[i], precision, recall, f_score])
        
    df = pd.DataFrame(f_scores, columns=["punctuation", "predicted", "predicted correctly", "predicted expectation", "precision", "recall", "f_score"])
    df = df.set_index("punctuation")
    return df, correct / total