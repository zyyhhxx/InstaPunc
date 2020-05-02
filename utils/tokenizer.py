from tqdm.auto import tqdm
from spacy.lang.en import English
import warnings
import string 

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
