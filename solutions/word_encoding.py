import numpy as np

def word_encoding(string):
    return np.mean([wordVectors[token] for token in clean_text(string) if token in wordVectors.vocab] or np.zeros((1, 300)), axis=0)

def encode_dataset(list_of_strings):
    X = np.empty((len(list_of_strings), wordVectors.vector_size))
    for i, s in enumerate(list_of_strings):
        X[i] = word_encoding(s)
    return X


X_train = encode_dataset(newsgroups_train.data)
X_test = encode_dataset(newsgroups_test.data)