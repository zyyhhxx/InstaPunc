from tqdm.auto import tqdm

def tokenize(sentences):
    from spacy.lang.en import English
    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    
    results = []
    for i in tqdm(range(len(sentences))):
        tokens = tokenizer(sentences[i])
        results.append(sentences[i])      
        
    return results