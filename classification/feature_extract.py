"""text feature extraction.
feature: word presence captured by bag of words."""

def bag_of_words(words):
    return dict([(word, True) for word in words])
