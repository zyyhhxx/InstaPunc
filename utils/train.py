import pandas as pd
from tqdm.auto import tqdm
from .constants import CLASSES
from .data_preprocessing import preprocess_data_inference
import torch

def train(train_loader, dev_loader, model, criterions, optimizer, epochs, device):

    for epoch in range(epochs):  # loop over the dataset multiple times
        print("----------------------------")
        print("Epoch:", epoch)

        model.train()
        t = tqdm(iter(train_loader), leave=False, total=len(train_loader))
        epoch_punctuation_loss = 0.0
        epoch_capitalization_loss = 0.0

        for _, data in enumerate(t, 0):
            inputs, punctuation_labels, capitalization_labels = \
                data[0].to(device), data[1][0].to(device), data[1][1].to(device)

            optimizer.zero_grad()
            predictions = model(inputs)
            punctuation_loss = criterions[0](predictions[0], punctuation_labels)
            capitalization_loss = criterions[1](predictions[1], capitalization_labels)

            loss = punctuation_loss + capitalization_loss

            loss.backward()
            optimizer.step()

            epoch_punctuation_loss += predictions[0].shape[0] * punctuation_loss.item()
            epoch_capitalization_loss += predictions[1].shape[0] * capitalization_loss.item()

        print("Training loss for punctuation:", epoch_punctuation_loss / len(train_loader))
        print("Training loss for capitalization:", epoch_capitalization_loss / len(train_loader))
        validate(dev_loader, model, device)
                
def eval(dataloader, model, device):
    punc_correct = 0
    cap_correct = 0
    total = 0
    punc_predicted_total = [1] * len(CLASSES)
    punc_predicted_correct = [0] * len(CLASSES)
    punc_predicted_expected = [1] * len(CLASSES)
    cap_predicted_total = [1] * 2
    cap_predicted_correct = [0] * 2
    cap_predicted_expected = [1] * 2
    
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, punctuation_labels, capitalization_labels = \
                data[0].to(device), data[1][0].to(device), data[1][1].to(device)

            predictions = model(inputs)
            _, punc_predicted = torch.max(predictions[0].data, 1)
            _, cap_predicted = torch.max(predictions[1].data, 1)

            # Gathering information for f score
            for i in range(punc_predicted.shape[0]):
                # For punctuation
                predicted_class = punc_predicted[i]
                correct_class = punctuation_labels[i]
                punc_predicted_total[predicted_class] += 1
                punc_predicted_expected[correct_class] += 1
                if predicted_class == correct_class:
                    punc_predicted_correct[predicted_class] += 1

                # For capitalization
                predicted_class = cap_predicted[i]
                correct_class = capitalization_labels[i]
                cap_predicted_total[predicted_class] += 1
                cap_predicted_expected[correct_class] += 1
                if predicted_class == correct_class:
                    cap_predicted_correct[predicted_class] += 1
            
            total += punc_predicted.shape[0]
            punc_correct += (punc_predicted == punctuation_labels).sum().item()
            cap_correct += (cap_predicted == capitalization_labels).sum().item()
    
    # Stats for punctuation
    f_scores = []
    for i in range(len(CLASSES)):
        precision = punc_predicted_correct[i] / punc_predicted_total[i]
        recall = punc_predicted_correct[i] / punc_predicted_expected[i]
        f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        f_scores.append([CLASSES[i], punc_predicted_total[i]-1, punc_predicted_correct[i], punc_predicted_expected[i]-1, precision, recall, f_score])
        
    df_punc = pd.DataFrame(f_scores, columns=["punctuation", "predicted", "predicted correctly", "predicted expectation", "precision", "recall", "f_score"])
    df_punc = df_punc.set_index("punctuation")

    # Stats for capitalization
    f_scores = []
    for i in range(2):
        precision = cap_predicted_correct[i] / cap_predicted_total[i]
        recall = cap_predicted_correct[i] / cap_predicted_expected[i]
        f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        f_scores.append([i, cap_predicted_total[i]-1, cap_predicted_correct[i], cap_predicted_expected[i]-1, precision, recall, f_score])
    f_scores[0][0] = "No"
    f_scores[1][0] = "Yes"
        
    df_cap = pd.DataFrame(f_scores, columns=["capitalization", "predicted", "predicted correctly", "predicted expectation", "precision", "recall", "f_score"])
    df_cap = df_cap.set_index("capitalization")

    return (df_punc, df_cap), (punc_correct / total, cap_correct / total)

def validate(dataloader, model, device):
    _, acc = eval(dataloader, model, device)
    print('Validation accuracy: punctuation: {}%, capitalization: {}%'.format(round(100 * acc[0], 4), round(100 * acc[1], 4)))

def test(dataloader, model, device):
    dfs, acc = eval(dataloader, model, device)
    print('Test accuracy: punctuation: {}%, capitalization: {}%'.format(round(100 * acc[0], 4), round(100 * acc[1], 4)))
    return dfs

def infer_sentence(sentence, model, vectors):
    model.eval()
    infer_x, token_data = preprocess_data_inference(sentence, vectors)
    
    predictions = model(infer_x)

    _, punc_predicted = torch.max(predictions[0].data, 1)
    _, cap_predicted = torch.max(predictions[1].data, 1)

    results = []
    for i in range(len(token_data)):
        if cap_predicted[i] == 1:
            results.append(token_data[i].capitalize())
        else:
            results.append(token_data[i])
        if punc_predicted[i] > 0:
            results.append(CLASSES[punc_predicted[i]])
    return " ".join(results)

def save_model(model, name = ""):
    now = datetime.now()
    PATH = './checkpoints/' + name + now.strftime("%m-%d-%Y-%H-%M-%S") + '.pth'
    torch.save(model.state_dict(), PATH)

def load_model(model, path):
    model.load_state_dict(torch.load(path))