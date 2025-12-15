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
    
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.base import BaseEstimator, TransformerMixin

class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        vector_size=300,
        window=8,
        min_count=2,
        epochs=40,
        dm=1,              # 1 = PV-DM, 0 = PV-DBOW
        workers=1,         # IMPORTANT for stability
        seed=42
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.dm = dm
        self.workers = workers
        self.seed = seed

    def fit(self, X, y=None):
        self.tagged_docs_ = [
            TaggedDocument(words=doc, tags=[i])
            for i, doc in enumerate(X)
        ]

        self.model_ = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            dm=self.dm,
            workers=self.workers,
            seed=self.seed
        )

        self.model_.build_vocab(self.tagged_docs_)
        self.model_.train(
            self.tagged_docs_,
            total_examples=len(self.tagged_docs_),
            epochs=self.epochs
        )

        return self

    def transform(self, X):
        return np.vstack([
            self.model_.infer_vector(doc, epochs=20)
            for doc in X
        ])

class TokenizerPreprocessor:
    def main_pipeline(self, text):
        return text.split()   # replace with your real tokenizer