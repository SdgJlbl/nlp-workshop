def words_to_sentiment(sentence):
    # Given a string representing a sentence, we want to split it into clean tokens with the clean_text function
    # Then we apply the vec_to_sentiment function to each token (if present in the vocabulary) to get a sentiment value
    # We return the average sentiment value
    return np.mean(vec_to_sentiment(np.array([wordVectors[w] for w in clean_text(sentence) if w in wordVectors.vocab])))
