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

from tqdm import tqdm  
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import numpy as np

def run_single_model_cv(
    model_instance, 
    model_name, 
    X, 
    y, 
    vectorizer, 
    dataset_name, 
    preprocessor=None, 
    n_splits=5, 
    random_state=42
):
    if preprocessor is None:
        preprocessor = IdentityPreprocessor()
        
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics_list = []

    print(f"\n{'='*20} TRAINING: {model_name} on {dataset_name} {'='*20}")

    # The loop uses standard 'tqdm' now
    for fold, (train_idx, test_idx) in enumerate(tqdm(mskf.split(X, y), total=n_splits, desc=f"Folds", leave=True), start=1):
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        
        if hasattr(y, "iloc"):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]

        modelhermetic = HermeticClassifier(
            preprocessor=preprocessor,
            vectorizer=vectorizer, 
            classifier=model_instance
        )

        modelhermetic.fit(X_train, y_train)

        y_train_pred = modelhermetic.predict(X_train)
        _, _, _, train_f1 = fold_score_calculator(y_train_pred, y_train, verbose=False)

        y_val_pred = modelhermetic.predict(X_test)
        val_acc, val_prec, val_rec, val_f1 = fold_score_calculator(y_val_pred, y_test, verbose=False)

        fold_metrics_list.append({
            "Train_F1": train_f1,
            "Val_F1": val_f1,
            "Val_Accuracy": val_acc,
            "Val_Precision": val_prec,
            "Val_Recall": val_rec
        })

    mean_metrics = pd.DataFrame(fold_metrics_list).mean()
    
    model_result_df = pd.DataFrame([{
        "Model": model_name,
        "Preprocessing": dataset_name,
        "Val_F1": mean_metrics["Val_F1"],
        "Train_F1": mean_metrics["Train_F1"],
        "Val_Accuracy": mean_metrics["Val_Accuracy"],
        "Val_Precision": mean_metrics["Val_Precision"],
        "Val_Recall": mean_metrics["Val_Recall"]
    }])
    
    print(f"Done! {model_name} [{dataset_name}] Val F1: {mean_metrics['Val_F1']:.4f}")
    
    return model_result_df
