"""
Evaluation Module
---------------------------------------------------

Functions and classes to evaluate model performance (e.g., accuracy, precision, recall, F1-score, confusion matrix visualization).
"""

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def compute_metrics(y_true, y_pred):
    pearson_r, _ = pearsonr(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true + 1, y_pred + 1)
    return {"Pearson R": round(pearson_r,3), "RMSE": round(rmse,3), "MAPE": round(mape,3)}