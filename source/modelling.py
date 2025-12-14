'''
Modelling Module
---------------------------------------------------

Reusable classes/functions for model training and prediction (e.g., a base Classifier class, functions to train common ML models).'''

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from gensim.models.doc2vec import TaggedDocument
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
from sklearn.base import clone



class HermeticClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, preprocessor, vectorizer, classifier, **kwargs):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.classifier = classifier

    def fit(self, X, y, **kwargs):

        X_preproc = [self.preprocessor.main_pipeline(doc) for doc in X]

        self.vectorizer_ = clone(self.vectorizer)
        X_train = self.vectorizer_.fit_transform(X_preproc)

        if hasattr(y, "to_numpy"):
            y_train = y.to_numpy()
        else:
            y_train = y

        # 4. Classification
        self.classifier.fit(X_train, y_train)

        self.classes_ = unique_labels(y) # Útil para checks do sklearn
        return self

    def predict(self, X, **kwargs):
        check_is_fitted(self, ['vectorizer_'])

        X_preproc = [self.preprocessor.main_pipeline(doc) for doc in X]
        X_test = self.vectorizer_.transform(X_preproc)
        
        return self.classifier.predict(X_test)
    

def fold_score_calculator(y_pred, y_test, verbose=False):
    
    #6. Compute the binary classification scores (accuracy, precision, recall, F1, AUC) for the fold.
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average="weighted")
    recall = metrics.recall_score(y_test, y_pred, average="weighted")
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")

    if verbose == True:
        print("Accuracy: {} \nPrecision: {} \nRecall: {} \nF1: {}".format(acc,prec,recall,f1))
    return (acc, prec, recall, f1)

    

class IdentityPreprocessor:
    def main_pipeline(self, text):
        return text