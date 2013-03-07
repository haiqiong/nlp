import collections
from feature_extract import bag_of_words
from nltk.classify import NaiveBayesClassifier
from nltk.probability import LaplaceProbDist
from nltk.classify.util import accuracy

"""
create {label:[featureset]} where featureset is a dict.
question: should all the words in a pos file be lableled pos?
"""

def get_labeled_data(corpus, feature_detector = bag_of_words):
    label_data = collections.defaultdict(list)
    for label in corpus.categories():
        for fileid in corpus.fileids(categories=[label]):
            features = feature_detector(corpus.words(fileids=[fileid]))
            label_data[label].append(features)
    return label_data

"""
divide labeled data as training set and test set.
"""
def split_label_data(label_data, split=0.75):
   train_set = []
   test_set = []
   for label, features in label_data.iteritems():
       index = int(len(features) * split)
       train_set.extend([(feature, label) for feature in features[:index]])
       test_set.extend([(feature, label) for feature in features[index:]])
   return train_set, test_set

"""
using Laplace estimator to construct prob distribution.
"""
def test(train_set, test_set):
    nb_classifier = NaiveBayesClassifier.train(train_set, 
                                               estimator=LaplaceProbDist)
    print(nb_classifier.classify(test_set))
    accuracy(nb_classifier, test_set)
        
        
