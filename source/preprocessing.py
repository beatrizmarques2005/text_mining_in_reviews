"""
Text Preprocessing Module
=========================

This module provides a flexible text preprocessing pipeline for text mining tasks,
such as classification, sentiment analysis, and co-occurrence analysis.

Functions
---------
1. regex_cleaner():
   Performs regex-based cleaning on raw text, removing noise such as emojis, hashtags, and URLs.

2. lemmatize_all():
   Lemmatizes a token using multiple parts of speech.

3. main_pipeline():
   Executes the full preprocessing workflow — cleaning, tokenization, stopword removal,
   normalization, lemmatization/stemming, and optional POS tagging.

4. cooccurrence_matrix_sentence_generator():
   Generates a co-occurrence matrix based on tokenized and preprocessed sentences.
"""

import re
import nltk
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer


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
    # Define patterns
    patterns = {
        "newline": r"(\n)",
        "hashtags_at": r"([#@])",
        "hashtags_ats_word": r"([#@]\w+)",
        "emojis": r"([\u2600-\u27FF])",
        "url": r"(?:\w+:/{2})?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(/[a-zA-Z/\d]+)?",
        "punctuation": r"[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff`]+",
        "apostrophe": r"'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'",
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

    return text.strip()


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
# MAIN PIPELINE
# ------------------------------------------------------------------------------
def main_pipeline(raw_text,
                  print_output=False,
                  no_stopwords=True,
                  custom_stopwords=None,
                  convert_diacritics=True,
                  lowercase=True,
                  lemmatized=True,
                  list_pos=["n", "v", "a", "r", "s"],
                  stemmed=False,
                  pos_tags_list="no_pos",
                  tokenized_output=False,
                  **kwargs):
    """
    Executes the main text preprocessing pipeline.

    Steps:
    1. Regex-based cleaning
    2. Tokenization
    3. Stopword removal
    4. Diacritic conversion
    5. Lemmatization / Stemming
    6. POS tagging (optional)
    7. Lowercasing (optional)
    8. Return either tokens or detokenized text

    Parameters
    ----------
    raw_text : str
        Input raw text.
    print_output : bool
        Print input and processed output.
    no_stopwords : bool
        Remove standard NLTK stopwords.
    custom_stopwords : list
        List of additional stopwords.
    convert_diacritics : bool
        Convert accented characters to ASCII.
    lowercase : bool
        Convert tokens to lowercase.
    lemmatized : bool
        Apply lemmatization.
    list_pos : list of str
        POS tags for lemmatization.
    stemmed : bool
        Apply stemming (Porter).
    pos_tags_list : str
        Options: "no_pos", "pos_list", "pos_tuples".
    tokenized_output : bool
        If True, return tokens instead of string.

    Returns
    -------
    list or str
        Preprocessed tokens or detokenized string.
    """
    if custom_stopwords is None:
        custom_stopwords = []

    text = regex_cleaner(raw_text, **kwargs)


    # --- Step: Tokenization and contraction handling ---
    tokens = nltk.tokenize.word_tokenize(text)

    # Apply contraction replacements before lemmatization
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

    # --- Step: Stopword removal ---
    if no_stopwords:
        base_stopwords = nltk.corpus.stopwords.words("english")

        # Optional: keep specific words (like 'again')
        keep_words = ["again"]
        stopwords = [w for w in base_stopwords if w not in keep_words]

        # Merge user-provided custom stopwords
        if custom_stopwords is None:
            custom_stopwords = []
        stopwords = set(stopwords + custom_stopwords)

        # Filter tokens
        tokens = [t for t in tokens if t.lower() not in stopwords]


    # Convert diacritics
    if convert_diacritics:
        tokens = [unidecode(t) for t in tokens]

    # Lemmatization and/or stemming
    if lemmatized:
        tokens = [lemmatize_all(t, list_pos) for t in tokens]
    if stemmed:
        stemmer = nltk.stem.PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    # POS tagging
    if pos_tags_list in {"pos_list", "pos_tuples"}:
        pos_tuples = nltk.pos_tag(tokens)
        if pos_tags_list == "pos_list":
            return [t[1] for t in pos_tuples]
        return pos_tuples

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
def cooccurrence_matrix_sentence_generator(preproc_sentences, sentence_cooc=False, window_size=5):
    """
    Builds a co-occurrence matrix from preprocessed tokenized sentences.

    Parameters
    ----------
    preproc_sentences : list of list of str
        List of tokenized sentences.
    sentence_cooc : bool
        If True, considers full sentence context; if False, uses a window size.
    window_size : int
        Size of the sliding context window.

    Returns
    -------
    pd.DataFrame
        Co-occurrence matrix (words as rows and columns).
    """
    co_occurrences = defaultdict(Counter)

    # Populate co-occurrence counts
    for sentence in tqdm(preproc_sentences, desc="Computing co-occurrences"):
        if sentence_cooc:
            for w1 in sentence:
                for w2 in sentence:
                    if w1 != w2:
                        co_occurrences[w1][w2] += 1
        else:
            for i, w1 in enumerate(sentence):
                for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                    if i != j:
                        co_occurrences[w1][sentence[j]] += 1

    # Create matrix
    vocab = sorted(set(word for sent in preproc_sentences for word in sent))
    word_idx = {w: i for i, w in enumerate(vocab)}
    matrix = np.zeros((len(vocab), len(vocab)), dtype=int)

    for w, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            matrix[word_idx[w]][word_idx[neighbor]] = count

    df = pd.DataFrame(matrix, index=vocab, columns=vocab)
    df = df.reindex(df.sum().sort_values(ascending=False).index, axis=0).reindex(df.sum().sort_values(ascending=False).index, axis=1)

    return df

