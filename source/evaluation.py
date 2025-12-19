"""
Evaluation Module
---------------------------------------------------

Functions and classes to evaluate model performance (e.g., accuracy, precision, recall, F1-score, confusion matrix visualization).
"""

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

def sentiment_analysis_metrics(y_true, y_pred):
    """
    Computes regression metrics commonly used for sentiment analysis or continuous
    score evaluation.

    Calculates Pearson Correlation, Root Mean Squared Error (RMSE), and 
    Mean Absolute Percentage Error (MAPE).
    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "Pearson R": Pearson correlation coefficient [-1, 1].
        - "RMSE": Root Mean Squared Error.
        - "MAPE": Mean Absolute Percentage Error (calculated with a +1 shift 
          to handle zero-values stability).
    """
    y_true, y_pred = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    pearson_r, _ = pearsonr(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true + 1, y_pred + 1)
    return {"Pearson R": round(pearson_r,3), "RMSE": round(rmse,3), "MAPE": round(mape,3)}
