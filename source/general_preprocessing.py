"""
Text Preprocessing Module
=========================

This module provides a flexible text preprocessing pipeline for text mining tasks,
such as classification, sentiment analysis, and co-occurrence analysis.

"""

# --- Standard Libraries ---
import re
from collections import defaultdict, Counter

# --- Data Handling ---
import numpy as np
import pandas as pd

# --- Progress Bars ---
from tqdm import tqdm

# --- NLP / Text Processing ---
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import words
from unidecode import unidecode

# --- Machine Learning / Feature Extraction ---
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# --- Visualization ---
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Language Detection ---
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import langid

# --- Translation ---
from deep_translator import GoogleTranslator

# ------------------------------------------------------------------------------
# REGEX CLEANER
# ------------------------------------------------------------------------------

def regex_cleaner(raw_text,
                  no_emojis=True,
                  no_hashtags=True,
                  hashtag_retain_words=True,
                  no_newlines=True,
                  no_urls=True,
                  no_punctuation=True):
    """
    Cleans raw text using regular expressions.

    Parameters
    ----------
    raw_text : str
        The input text to clean.
    no_emojis : bool, optional
        Remove emojis from text.
    no_hashtags : bool, optional
        Remove hashtags and @mentions.
    hashtag_retain_words : bool, optional
        Retain words in hashtags if True (e.g., "#great" → "great").
    no_newlines : bool, optional
        Replace newlines with spaces.
    no_urls : bool, optional
        Remove URLs.
    no_punctuation : bool, optional
        Remove punctuation, keeping apostrophes that aid tokenization.

    Returns
    -------
    str
        Cleaned text.
    """

    patterns = {
        "newline": r"(\n)",
        "hashtags_at": r"([#@])",
        "hashtags_ats_word": r"([#@]\w+)",
        "emojis": r"([\u2600-\u27FF])",
        "url": r"(?:\w+:/{2})?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(/[a-zA-Z/\d]+)?",
        "punctuation": r"[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff`]+",
        "apostrophe": r"'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'",
        "hifen": r'\s*-\s*',
    }

    text = raw_text

    if no_emojis:
        text = re.sub(patterns["emojis"], "", text)
    if no_hashtags:
        if hashtag_retain_words:
            text = re.sub(patterns["hashtags_at"], "", text)
        else:
            text = re.sub(patterns["hashtags_ats_word"], "", text)
    if no_newlines:
        text = re.sub(patterns["newline"], " ", text)
    if no_urls:
        text = re.sub(patterns["url"], "", text)
    if no_punctuation:
        text = re.sub(patterns["punctuation"], "", text)
        text = re.sub(patterns["apostrophe"], "", text)
        text = re.sub(patterns["hifen"], ' ', text) # no hifens

    return text.strip()

def repeated_chars(token, max_repeat=2):
    """
    Reduces consecutive repeated characters beyond `max_repeat` to that limit.
    Example:
        'soooo' -> 'soo'
        'niiiiceeee' -> 'niicee'
        'excellent' -> 'excellent'
    """
    return re.sub(r'(.)\1{%d,}' % max_repeat, r'\1' * max_repeat, token)

# ------------------------------------------------------------------------------
# LEMMATIZATION
# ------------------------------------------------------------------------------

def lemmatize_all(token, list_pos=["n", "v", "a", "r", "s"]):
    """
    Lemmatizes a token using multiple POS tags.

    Parameters
    ----------
    token : str
        Input token.
    list_pos : list of str
        POS tags to attempt lemmatization for (default: noun, verb, adjective, adverb, satellite adj).

    Returns
    -------
    str
        Lemmatized token.
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for pos in list_pos:
        token = lemmatizer.lemmatize(token, pos)
    return token

# ------------------------------------------------------------------------------
# VECTORIZATION
# ------------------------------------------------------------------------------

def vectorize_texts(text, vectorizer_type="tfidf", max_features=1000):

    # Convert tokenized lists to strings
    processed_texts = pd.Series(text).apply(
        lambda x: " ".join(x) if isinstance(x, (list, tuple)) else str(x)
    )

    # Choose vectorizer
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features)
    elif vectorizer_type == "count":
        vectorizer = CountVectorizer(max_features=max_features)
    else:
        raise ValueError("Invalid vectorizer_type. Choose 'tfidf' or 'count'.")

    dtm = vectorizer.fit_transform(processed_texts)
    return dtm, vectorizer

'''Frequency & One-hot encoding 
TF-IDF 
N-gram Vectorizer
Bag of Words '''


# ------------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------------

def main_pipeline(raw_text,
                  print_output=False,
                  no_stopwords=True,
                  stopwords_tokeep=None,
                  extra_stopwords=None,
                  convert_diacritics=True,
                  lowercase=True,
                  lemmatized=True,
                  list_pos=["n", "v", "a", "r", "s"],
                  stemmed=False,
                  pos_tags_list="no_pos",
                  tokenized_output=False,
                  treat_repeated_chars=False,
                  **kwargs):
    """
    Executes the main text preprocessing pipeline.

    Steps (grouped):
    1. Clean data      - remove extraneous content.
    2. Transform data  - tokenization, POS tagging, stopwords removal.
    3. Normalize data  - case, stemming, and lemmatization.

    Other docstring details omitted for brevity.
    """

    if extra_stopwords is None:
        extra_stopwords = []
    
    if stopwords_tokeep is None:
        stopwords_tokeep = []

    # --------------------------------------------------------------------------
    # PART 1: CLEAN DATA - remove extraneous content (regex cleaning, basic prep)
    # --------------------------------------------------------------------------
    text = regex_cleaner(raw_text, **kwargs)

    # --------------------------------------------------------------------------
    # PART 2: TRANSFORM DATA - tokenization, contraction handling, POS & stopwords
    # --------------------------------------------------------------------------
    # Tokenization
    tokens = nltk.tokenize.word_tokenize(text)

    # Contraction replacements (handled as part of transformation)
    contraction_map = {
        "'m": "am",
        "n't": "not",
        "'s": "is",
        "'re": "are",
        "'ve": "have",
        "'ll": "will",
        "'d": "would"
    }

    normalized_tokens = []
    for tok in tokens:
        new_tok = tok
        for pattern, repl in contraction_map.items():
            new_tok = re.sub(pattern, repl, new_tok)
        normalized_tokens.append(new_tok)
    tokens = normalized_tokens

    # Optional repeated-character normalization (transformation step can include token-level cleanup)
    if treat_repeated_chars:
        tokens = [repeated_chars(t) for t in tokens]

    # Stopword removal
    if no_stopwords:
        base_stopwords = nltk.corpus.stopwords.words("english")
        stopwords = [w for w in base_stopwords if w not in stopwords_tokeep]
        stopwords = set(stopwords + extra_stopwords)
        tokens = [t for t in tokens if t.lower() not in stopwords]

    # POS tagging (if requested, return early according to mode)
    if pos_tags_list in {"pos_list", "pos_tuples"}:
        pos_tuples = nltk.pos_tag(tokens)
        if pos_tags_list == "pos_list":
            return [t[1] for t in pos_tuples]
        return pos_tuples

    # --------------------------------------------------------------------------
    # PART 3: NORMALIZE DATA - diacritics, lemmatization, stemming, case
    # --------------------------------------------------------------------------
    # Convert diacritics (accented -> ascii)
    if convert_diacritics:
        tokens = [unidecode(t) for t in tokens]

    # Lemmatization and/or stemming
    if lemmatized:
        tokens = [lemmatize_all(t, list_pos) for t in tokens]
    if stemmed:
        stemmer = nltk.stem.PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    # Lowercasing
    if lowercase:
        tokens = [t.lower() for t in tokens]

    if print_output:
        print("Raw:", raw_text)
        print("Processed:", tokens)

    return tokens if tokenized_output else TreebankWordDetokenizer().detokenize(tokens)

# ------------------------------------------------------------------------------
# CO-OCCURRENCE MATRIX
# ------------------------------------------------------------------------------

def cooccurrence_matrix(vectorized_df, sentence_cooc=False, window_size=5):
    """
    Builds a co-occurrence matrix from a document-term DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Document-term matrix (documents as rows, words as columns). Must be numeric.
    sentence_cooc : bool
        If True, considers full sentence context; if False, uses a sliding window (currently not used for DataFrame input).
    window_size : int
        Size of sliding window (ignored for DataFrame input).

    Returns
    -------
    pd.DataFrame
        Co-occurrence matrix.
    """
    # Ensure numeric values
    X_dense = vectorized_df.astype(float).values
    feature_names = vectorized_df.columns.tolist()
    n_words = len(feature_names)
    
    co_matrix = np.zeros((n_words, n_words), dtype=int)
    word_idx = {w: i for i, w in enumerate(feature_names)}

    # Co-occurrence by document
    for doc_vector in tqdm(X_dense, desc="Computing co-occurrences"):
        present_indices = np.where(doc_vector > 0)[0]
        for i in present_indices:
            for j in present_indices:
                if i != j:
                    co_matrix[i, j] += 1

    cooc_df = pd.DataFrame(co_matrix, index=feature_names, columns=feature_names)
    cooc_df = cooc_df.reindex(cooc_df.sum().sort_values(ascending=False).index, axis=0)\
                     .reindex(cooc_df.sum().sort_values(ascending=False).index, axis=1)
    
    return cooc_df

#---------------------------------------------------------------------------------------------------
#                                       NON-ENGLISH-REVIEWS
#---------------------------------------------------------------------------------------------------

# Set seed for reproducibility
DetectorFactory.seed = 0

def extract_non_english_reviews_langdetect(df, text_column="review"):
    """
    Detects the language of each review and returns a DataFrame
    containing only non-English reviews with their original indices.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing text data.
    text_column : str, default='review'
        Name of the column that holds the review text.

    Returns
    -------
    non_english_df : pd.DataFrame
        DataFrame with columns: ['index', text_column, 'language']
        containing only non-English reviews.
    """

    # Inner function for safe detection
    def detect_language(text):
        try:
            return detect(str(text))
        except LangDetectException:
            return "unknown"

    # Detect language for each review
    df["language"] = df[text_column].apply(detect_language)

    # Filter non-English rows
    non_english_df = df[df["language"] != "en"].copy()

    # Keep original index for reference
    non_english_df = non_english_df.reset_index()[["index", text_column, "language"]]

    return non_english_df


def extract_non_english_reviews_langid(df, text_column="review"):
    """
    Detects the language of each review using langid and returns
    non-English reviews with their original indices.
    """
    def detect_language(text):
        lang, _ = langid.classify(str(text))
        return lang
    #def detect_language(text, min_prob=0.85):
    #    text = str(text)
    #    lang, prob = langid.classify(text)
    #    if prob < min_prob:
    #        return "en"
    #    return lang
    # Detect language
    df["language"] = df[text_column].apply(detect_language)

    # Filter non-English rows
    non_english_df = df[df["language"] != "en"].copy()
    non_english_df = non_english_df.reset_index()[["index", text_column, "language"]]

    return non_english_df

#---------------------------------------------------------------------------------------------------
#                                       TRANSLATION TO ENGLISH
#---------------------------------------------------------------------------------------------------

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text  # fallback if translation fails