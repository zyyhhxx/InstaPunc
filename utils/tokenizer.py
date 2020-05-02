from tqdm.auto import tqdm
from spacy.lang.en import English
import warnings
import string 

def preprocess_data(dataset, window_size, classes):
    token_data = tokenize(dataset)
    padded_data = pad(token_data, window_size)
    data, labels = create_labels(padded_data, classes, window_size)

    return data, labels

def tokenize(sentences):
    print("Tokenizing:")

    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    results = []

    for i in tqdm(range(len(sentences))):
        # Skip empty sentences
        if len(sentences[i]) <= 0:
            continue
        
        tokens = tokenizer(sentences[i])
        results.append([token.text.lower() for token in tokens])      
        
    return results

def pad(sentences, window_size):
    print("Padding:")

    pad_size = window_size // 2
    results = []

    for i in tqdm(range(len(sentences))):
        results.append(["<pad>"] * pad_size + sentences[i] + ["<pad>"] * pad_size)
    
    return results


def create_labels(sentences, classes, window_size):
    """ 
    Create labels based on classes
  
    Parameters: 
    sentences (string[]): tokenized text
    classes (string[]): classes of punctuation
    window_size (int): the size of the window, should be an odd number
  
    Returns: 
    string[]: the text without punctuation
    string[]: the labels corresponding the punctuation in the middle of the n-gram data
  
    """

    print("Creating labels:")

    pad_size = window_size // 2
    data = []
    labels = []

    for i in tqdm(range(len(sentences))):
        sentence = sentences[i]

        # First, get labels
        sentence_labels = []
        pre_punctuation = False
        pre_pad = True

        for j in range(pad_size, len(sentence) - pad_size + 1):
            word = sentence[j]

            if len(word) == 1 and word in string.punctuation:
                # If the word is a class punctuation, insert it regardless
                if word in classes:
                    sentence_labels.append(word)
                    pre_punctuation = True
            else:
                # Only add "no punctuation" when the previous word is not punctuation or ignored (including padding) and 
                if not pre_punctuation and not pre_pad:
                    # o means no punctuation
                    sentence_labels.append("o")
                pre_punctuation = False
                pre_pad = False

        # Second, get tokens without punctuations
        clean_sentence = []
        for word in sentence:
            if len(word) == 1 and word in string.punctuation:
                continue
            else:
                clean_sentence.append(word)

        if len(sentence_labels) != len(clean_sentence) - window_size + 1:
            warnings.warn("Lengths of labels and non-punctuation words mismatch:" + str(i))

        # Third, construct data
        for j in range(len(clean_sentence) - window_size + 1):
            data.append(clean_sentence[j:j + window_size])
            labels.append(sentence_labels[j])

    return data, labels

def create_labels_single(sentence, classes, window_size):
    data = []
    labels = []
    
    pad_size = window_size // 2

    # First, get labels
    sentence_labels = []
    pre_punctuation = False
    pre_pad = True

    for j in range(pad_size, len(sentence) - pad_size + 1):
        word = sentence[j]

        if len(word) == 1 and word in string.punctuation:
            # If the word is a class punctuation, insert it regardless
            if word in classes:
                sentence_labels.append(word)
                pre_punctuation = True
        else:
            # Only add "no punctuation" when the previous word is not punctuation or ignored (including padding) and 
            if not pre_punctuation and not pre_pad:
                # o means no punctuation
                sentence_labels.append("o")
            pre_punctuation = False
            pre_pad = False

        print(word, "".join(sentence_labels), len(sentence_labels))

    # Second, get tokens without punctuations
    clean_sentence = []
    for word in sentence:
        if len(word) == 1 and word in string.punctuation:
            continue
        else:
            clean_sentence.append(word)

    # Third, construct data
    for j in range(len(clean_sentence) - window_size + 1):
        data.append(clean_sentence[j:j + window_size])
        labels.append(sentence_labels[j])

    # Finally, inspect the results
    for i in range(len(data)):
        print(data[i], labels[i])