from tqdm.auto import tqdm
from spacy.lang.en import English
import warnings
import string
import torch
from .constants import CLASSES, WINDOW_SIZE

def preprocess_data(dataset, vectors):
    token_data = tokenize(dataset)
    padded_data = pad(token_data, WINDOW_SIZE)
    x_range, labels = create_labels(padded_data, CLASSES, WINDOW_SIZE)

    x_data = get_word_vector(padded_data, vectors)
    y = convert_labels(labels)

    print("Printing samples of the dataset")
    for i in range(15):
        data_index, data_left, data_right = x_range[i]
        print(padded_data[data_index][data_left:data_right], labels[i])

    return (x_range, x_data), y

def preprocess_data_inference(dataset, vectors):
    token_data = tokenize([dataset], False)
    padded_data = pad(token_data, WINDOW_SIZE, False)
    data = get_n_gram(padded_data[0], WINDOW_SIZE)
    x = get_word_vector(data, vectors, False)
    
    return x, token_data[0]

def tokenize(sentences, progress=True):
    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    results = []

    if progress:
        print("Tokenizing:")
        for i in tqdm(range(len(sentences))):
            # Skip empty sentences
            if len(sentences[i]) <= 1:
                continue
            
            tokens = tokenizer(sentences[i])
            results.append([token.text for token in tokens])      

    else:
        for i in range(len(sentences)):
            # Skip empty sentences
            if len(sentences[i]) <= 1:
                continue
            
            tokens = tokenizer(sentences[i])
            results.append([token.text for token in tokens])
        
    return results

def pad(sentences, window_size, progress=True):
    pad_size = window_size // 2
    results = []

    if progress:
        print("Padding:")
        for i in tqdm(range(len(sentences))):
            results.append(["<pad>"] * pad_size + sentences[i] + ["<pad>"] * pad_size)

    else:
        for i in range(len(sentences)):
            results.append(["<pad>"] * pad_size + sentences[i] + ["<pad>"] * pad_size)
    
    return results

def create_labels(sentences, classes, window_size, progress=True):
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

    pad_size = window_size // 2
    data = []
    labels = []

    if progress:
        print("Creating labels:")
        for i in tqdm(range(len(sentences))):
            sentence = sentences[i]

            # First, get punctuation labels
            punc_labels = get_punc_labels(sentence, pad_size, classes)

            # Second, get tokens without punctuations
            clean_sentence = get_clean_sentence(sentence, punc_labels, window_size)

            # Third, construct data
            data += get_n_gram_range(clean_sentence, window_size, i)
            get_labels(labels, clean_sentence, window_size, punc_labels)

    else:
        for i in range(len(sentences)):
            sentence = sentences[i]

            # First, get punctuation labels
            punc_labels = get_punc_labels(sentence, pad_size, classes)

            # Second, get tokens without punctuations
            clean_sentence = get_clean_sentence(sentence, punc_labels, window_size)

            # Third, construct data
            data += get_n_gram_range(clean_sentence, window_size, i)
            get_labels(labels, clean_sentence, window_size, punc_labels)

    return data, labels

def get_punc_labels(sentence, pad_size, classes):
    sentence_labels = []
    pre_punctuation = False
    pre_pad = True

    for j in range(pad_size, len(sentence) - pad_size + 1):
        word = sentence[j]

        if len(word) == 1 and word in string.punctuation:
            # Handling special cases of more than one consecutive punctuations
            if pre_punctuation:
                # For handling etc.
                if word == "," and sentence[j-1] == '.' :
                    sentence_labels[-1] = ","

                # Ignoring all consecutive punctuations except the very first one
                else:
                    continue

            # If the word is a class punctuation, insert it regardless
            elif word in classes and not pre_pad:
                sentence_labels.append(word)
                pre_punctuation = True
        else:
            # Only add "no punctuation" when the previous word is not punctuation or ignored (including padding) and 
            if not pre_punctuation and not pre_pad:
                # o means no punctuation
                sentence_labels.append("o")
            pre_punctuation = False
            pre_pad = False

    return sentence_labels

def get_clean_sentence(sentence, punc_labels, window_size):
    clean_sentence = []

    for word in sentence:
        if len(word) == 1 and word in string.punctuation:
            continue
        else:
            clean_sentence.append(word)

    if len(punc_labels) != len(clean_sentence) - window_size + 1:
        warnings.warn("Lengths of labels and non-punctuation words mismatch:" + str(i))
        # Test the troubled sentence automatically
        # create_labels_test(sentence, classes, window_size)

    return clean_sentence

def get_n_gram_range(clean_sentence, window_size, sentence_index):
    results = []

    for j in range(len(clean_sentence) - window_size + 1):
        results.append((sentence_index, j, j + window_size))

    return results

def get_labels(labels, clean_sentence, window_size, punc_labels):
    for j in range(len(clean_sentence) - window_size + 1):
        n_gram_data = clean_sentence[j:j + window_size]
        punctuation_label = punc_labels[j]
        capitalization_label = n_gram_data[len(n_gram_data)//2][0].isupper()
        labels.append((punctuation_label, capitalization_label))

def create_labels_test(sentence, classes, window_size):
    print("Testing: " + " ".join(sentence))

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
            # Handling special cases of more than one consecutive punctuations
            if pre_punctuation:
                # For handling etc.
                if word == "," and sentence[j-1] == '.' :
                    sentence_labels[-1] = ","

                # Ignoring all consecutive punctuations except the very first one
                else:
                    continue

            # If the word is a class punctuation, insert it regardless
            elif word in classes and not pre_pad:
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

def get_word_vector(data, word_vector, progress=True):
    word_vector_weights= []
    
    if progress:
        print("Get word vector weights")
        for i in tqdm(range(len(data))):
            temp_weights = []
            
            for word in data[i]:
                temp_weights.append(word_vector[word.lower()])
            
            word_vector_weights.append(torch.stack(temp_weights))

    else:
        for i in range(len(data)):
            temp_weights = []
            
            for word in data[i]:
                temp_weights.append(word_vector[word.lower()])
            
            word_vector_weights.append(torch.stack(temp_weights))

    return word_vector_weights

def convert_labels(labels, progress=True):
    punctuation_encodings = []
    capitalization_encodings = []
    classes = {}
    class_num = 0

    if progress:
        print("Converting labels to tensor")
        for i in tqdm(range(len(labels))):
            punctuation_label, capitalization_label = labels[i]

            capitalization_encodings.append(int(capitalization_label))

            if punctuation_label in classes:
                punctuation_encodings.append(classes[punctuation_label])
            else:
                classes[punctuation_label] = CLASSES.index(punctuation_label)
                class_num += 1
                punctuation_encodings.append(classes[punctuation_label])

    else:
        for i in range(len(labels)):
            punctuation_label, capitalization_label = labels[i]

            capitalization_encodings.append(int(capitalization_label))

            if punctuation_label in classes:
                punctuation_encodings.append(classes[punctuation_label])
            else:
                classes[punctuation_label] = CLASSES.index(punctuation_label)
                class_num += 1
                punctuation_encodings.append(classes[punctuation_label])

    return torch.LongTensor(punctuation_encodings), torch.LongTensor(capitalization_encodings)
