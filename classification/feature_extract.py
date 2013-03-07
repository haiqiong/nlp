"""text feature extraction.
feature: word presence captured by bag of words."
1. filter stopwords.
2. include significant bigrams using chi_sq score function.
"""
import nltk
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

def bag_of_words(words, stopfile='english', score_fn=BigramAssocMeasures.chi_sq,
                 n=200):
    remaining_words = set(words) - set(stopwords.words(stopfile))
    return dict([(word, True) for word in (remaining_words)])
    """
    bigram_finder = BigramCollocationFinder.from_words(remaining_words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(word, True) for word in (remaining_words | set(bigrams))])
    """
