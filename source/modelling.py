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



class HermeticClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, preprocessor, vectorizer, classifier, **kwargs):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.classifier = classifier

    def fit(self, X, y, **kwargs):

        X_preproc = [self.preprocessor.main_pipeline(doc) for doc in X]

        self.vectorizer_ = self.vectorizer
        X_train = self.vectorizer.fit_transform(X_preproc)

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
        X_test = self.vectorizer.transform(X_preproc)
        
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Binary cross entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p = probability after sigmoid
        probas = torch.sigmoid(logits)

        # focal factor
        focal_factor = (1 - probas) ** self.gamma

        loss = self.alpha * focal_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
    

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)

class MultiLabelBERT(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls)
        return logits

