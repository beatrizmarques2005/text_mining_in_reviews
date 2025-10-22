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
    if type(user_review) == list:
        sent_compound_list = []
        for sentence in user_review:
            sent_compound_list.append(vader.polarity_scores(sentence)["compound"])
        polarity = np.array(sent_compound_list).mean()
    else:
        polarity = vader.polarity_scores(user_review)["compound"]
    return polarity


def textblob_wrapper(user_review):   
    if type(user_review) == list:
        sent_compound_list = []
        for sentence in user_review:
            sent_compound_list.append(TextBlob(sentence).sentiment.polarity)
        polarity = np.array(sent_compound_list).mean()
    else:
        polarity = TextBlob(user_review).sentiment.polarity
    return polarity