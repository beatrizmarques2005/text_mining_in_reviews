'''
Modelling Module
---------------------------------------------------

Reusable classes/functions for model training and prediction (e.g., a base Classifier class, functions to train common ML models).'''

# =============================================================================
# IMPORTS
# =============================================================================

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from gensim.models.doc2vec import TaggedDocument
from sklearn.base import clone
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
from gensim.models import Doc2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# =============================================================================
# CLASSES
# =============================================================================

class HermeticClassifier(ClassifierMixin, BaseEstimator):
    """
    A wrapper class that ensures total isolation of the vectorization process 
    within the cross-validation fold to prevent data leakage.

    It fits the vectorizer only on the training data passed to `fit()`, 
    and then transforms the test data in `predict()`.
    """
    def __init__(self, vectorizer, classifier, **kwargs):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def fit(self, X, y, **kwargs):

        X_preproc = X

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

        X_preproc = X
        X_test = self.vectorizer_.transform(X_preproc)
        
        return self.classifier.predict(X_test)
    
class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    A Scikit-Learn compatible wrapper for Gensim's Doc2Vec model.
    Allows Doc2Vec to be used in sklearn Pipelines and cross-validation loops.
    """
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
    """
    A placeholder class for text preprocessing and tokenization.
    """
    def main_pipeline(self, text):
        """
        Splits text into tokens.
        Returns
        -------
        list
            A list of string tokens.
        """
        return text.split() 

# =============================================================================
# EVALUATION & UTILITY FUNCTIONS
# =============================================================================

def fold_score_calculator(y_pred, y_test, verbose=False):
    """
    Calculates standard classification metrics for a single fold.

    Returns
    -------
    tuple
        A tuple containing (accuracy, precision_weighted, recall_weighted, f1_weighted).
    """
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average="weighted")
    recall = metrics.recall_score(y_test, y_pred, average="weighted")
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")

    if verbose == True:
        print("Accuracy: {} \nPrecision: {} \nRecall: {} \nF1: {}".format(acc,prec,recall,f1))
    return (acc, prec, recall, f1)


def run_single_model_cv(
    model_instance, 
    model_name, 
    X, 
    y, 
    vectorizer, 
    dataset_name,
    n_splits=5, 
    random_state=42
):
    """
    Runs Multilabel Stratified Cross-Validation for a single model configuration.
    
    This function handles the training loop, 'hermetic' vectorization, and 
    metric aggregation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the averaged validation metrics across all folds.
    """   
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics_list = []

    print(f"\n{'='*20} TRAINING: {model_name} on {dataset_name} {'='*20}")

    for fold, (train_idx, test_idx) in enumerate(tqdm(mskf.split(X, y), total=n_splits, desc=f"Folds", leave=True), start=1):
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        
        if hasattr(y, "iloc"):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]

        modelhermetic = HermeticClassifier(
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


def plot_top_features(model, vocabulary, class_labels, top_n=8):
    """
    Visualizes the top positive and negative coefficients for a linear model.
    
    Designed for One-vs-Rest (OvR) wrappers where `model.estimators_` contains
    a list of binary classifiers.

    Returns
    -------
    None
        Displays a matplotlib plot.
    """
    n_classes = len(class_labels)
    ncols = 3
    nrows = (n_classes // ncols) + (1 if n_classes % ncols > 0 else 0)
   
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
    axes = axes.flatten()
 
    for i, label in enumerate(class_labels):

        coefs = model.estimators_[i].coef_.flatten()
  
        top_positive_indices = coefs.argsort()[-top_n:]
        top_negative_indices = coefs.argsort()[:top_n]
       
        top_indices = list(top_positive_indices) + list(top_negative_indices)
        top_coefs = coefs[top_indices]
        top_words = [vocabulary[j] for j in top_indices]

        ax = axes[i]
        colors = ['red' if c < 0 else 'blue' for c in top_coefs]
        ax.barh(top_words, top_coefs, color=colors)
        ax.set_title(f"{label}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Weight")

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
       
    plt.tight_layout()
    plt.show()

def is_false_negative(row):
    """
    Checks if a specific row represents a False Negative for a global target category.

    Returns
    -------
    bool
        True if the target category is in True_Labels but not in Predicted_Labels.
    """
    return (target_category in row['True_Labels']) and (target_category not in row['Predicted_Labels'])


def compute_label_language_similarity(
    X,
    y_df,
    max_df=0.8,
    min_df=5
):
    """
    Computes a label-label cosine similarity matrix based on TF-IDF
    representations of review text associated with each label.

    Returns
    -------
    similarity_df : pd.DataFrame
        Label-label cosine similarity matrix.
    """

    assert len(X) == len(y_df)
    assert X.index.equals(y_df.index)

    label_texts = []

    for label in y_df.columns:
        texts_for_label = X.loc[y_df[label] == 1]

        # Handle rare / empty labels safely
        combined_text = " ".join(texts_for_label) if len(texts_for_label) > 0 else ""

        label_texts.append({
            "label": label,
            "text": combined_text
        })

    tfidf_vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(
        [item["text"] for item in label_texts]
    )


    similarity_matrix = cosine_similarity(tfidf_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=[item["label"] for item in label_texts],
        columns=[item["label"] for item in label_texts]
    )

    return similarity_df


def merge_labels(y_df, merges):
    """
    Merges multiple binary columns into a single column (Logical OR) and 
    drops the original columns.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with merged columns.
    """
    y_df = y_df.copy()

    for new_label, old_labels in merges.items():
        y_df[new_label] = y_df[old_labels].any(axis=1).astype(int)

        y_df = y_df.drop(columns=old_labels)

    return y_df

def evaluate_model_cv(X, y, classifier, vectorizer, cv, mlb, preprocessor, wrapper_class):
    """
    Performs a comprehensive cross-validation evaluation of a model, providing
    both global and per-category performance metrics.

    Returns
    -------
    global_avg : pd.DataFrame
        Average global metrics (F1, Accuracy, Precision, Recall) across folds.
    cat_avg : pd.DataFrame
        Detailed metrics broken down by category, including overfitting gap.
    """
    rows_categories = []
    rows_global = []

    print(f"Running full analysis on: {type(classifier).__name__}...")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):

        if hasattr(X, "iloc"):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]
            
        if hasattr(y, "iloc"):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]

        hermetic_model = wrapper_class(
            vectorizer=vectorizer,
            classifier=classifier
        )

        hermetic_model.fit(X_train, y_train)
        y_val_pred = hermetic_model.predict(X_test)
        y_train_pred = hermetic_model.predict(X_train)

        val_accuracy = accuracy_score(y_test, y_val_pred)
        
        val_report = classification_report(
            y_test, y_val_pred, target_names=mlb.classes_, zero_division=0, output_dict=True
        )
        train_report = classification_report(
            y_train, y_train_pred, target_names=mlb.classes_, zero_division=0, output_dict=True
        )

        for label in mlb.classes_:
            rows_categories.append({
                "Fold": fold,
                "Category": label,
                "Train_F1": train_report[label]["f1-score"],
                "Val_F1": val_report[label]["f1-score"],
                "Val_Precision": val_report[label]["precision"],
                "Val_Recall": val_report[label]["recall"],
                "Support": val_report[label]["support"]
            })

        rows_global.append({
            "Fold": fold,
            "Val_Accuracy": val_accuracy,
            "Val_Weighted_Precision": val_report["weighted avg"]["precision"],
            "Val_Weighted_Recall": val_report["weighted avg"]["recall"],
            "Val_Weighted_F1": val_report["weighted avg"]["f1-score"],
            "Train_Weighted_F1": train_report["weighted avg"]["f1-score"]
        })

    df_global = pd.DataFrame(rows_global)
    global_avg = df_global.mean().to_frame(name="Average Score").T.drop(columns=["Fold"])
    
    cols_order = ["Train_Weighted_F1", "Val_Weighted_F1", "Val_Weighted_Precision", "Val_Weighted_Recall", "Val_Accuracy"]

    global_avg = global_avg[[c for c in cols_order if c in global_avg.columns]]

    print("\n" + "="*40)
    print(" 🌍 GLOBAL MODEL PERFORMANCE (Avg 5-Folds)")
    print("="*40)
    try:
        display(global_avg)
    except NameError:
        print(global_avg)

    df_categories = pd.DataFrame(rows_categories)
    cat_avg = df_categories.groupby("Category")[
        ["Train_F1", "Val_F1", "Val_Precision", "Val_Recall", "Support"]
    ].mean()

    cat_avg["Overfit_Gap"] = cat_avg["Train_F1"] - cat_avg["Val_F1"]
    cat_avg = cat_avg.sort_values("Val_F1", ascending=False)

    print("\n" + "="*40)
    print(" DETAILED CATEGORY BREAKDOWN")
    print("="*40)
    try:
        display(cat_avg)
    except NameError:
        print(cat_avg)
        
    return global_avg, cat_avg