"""
Sentiment Analysis Preparation Module
---------------------------------------------------

This module contains functions and classes to preprocess and prepare text data for sentiment analysis tasks.

"""
# =============================================================================
# IMPORTS
# =============================================================================

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
vader = SentimentIntensityAnalyzer()

# =============================================================================
# VADER WRAPPER FUNCTION
# =============================================================================

def vader_wrapper(user_review):
    """
    Returns a dict with keys: 'compound', 'pos', 'neg', 'neu'.
    If user_review is a list, returns the mean of each score across sentences.
    If user_review is a single string, returns VADER's scores for that string.
    """
    if isinstance(user_review, list):
        pos_list, neg_list, neu_list, comp_list = [], [], [], []
        for sentence in user_review:
            scores = vader.polarity_scores(sentence)
            pos_list.append(scores["pos"])
            neg_list.append(scores["neg"])
            neu_list.append(scores["neu"])
            comp_list.append(scores["compound"])
        return {
            "compound": float(np.mean(comp_list)),
            "pos": float(np.mean(pos_list)),
            "neg": float(np.mean(neg_list)),
            "neu": float(np.mean(neu_list)),
        }
    else:
        scores = vader.polarity_scores(user_review)
        return {k: float(scores[k]) for k in ("compound", "pos", "neg", "neu")}


def textblob_wrapper(user_review):
    """
    Returns a dict with keys: 'polarity' and 'subjectivity'.
    
    - If user_review is a list, returns the mean polarity and subjectivity across sentences.
    - If user_review is a single string, returns TextBlob's scores directly.
    """
    if isinstance(user_review, list):
        polarities = [TextBlob(s).sentiment.polarity for s in user_review]
        subjectivities = [TextBlob(s).sentiment.subjectivity for s in user_review]
        return {
            "polarity": float(np.mean(polarities)) if polarities else 0.0,
            "subjectivity": float(np.mean(subjectivities)) if subjectivities else 0.0,
        }
    else:
        sentiment = TextBlob(user_review).sentiment
        return {
            "polarity": float(sentiment.polarity),
            "subjectivity": float(sentiment.subjectivity),
        }

