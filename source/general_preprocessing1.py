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
import emoji
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import words
from unidecode import unidecode

# --- Machine Learning / Feature Extraction ---
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator

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

# --- Optional Safe Imports ---
try:
    from langdetect import detect as ld_detect
except Exception:
    ld_detect = None

try:
    import langid
except Exception:
    langid = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None


# ------------------------------------------------------------------------------
# REGEX CLEANER
# ------------------------------------------------------------------------------

def regex_cleaner(raw_text,
                  no_emojis=True,
                  no_hashtags=True,
                  hashtag_retain_words=True,
                  no_newlines=True,
                  no_urls=True,   # <-- new option
                  no_punctuation=True):
    """
    Cleans raw text using regular expressions.
    """

    patterns = {
        "newline": r"(\n)",
        "hashtags_at": r"([#@])",
        "hashtags_ats_word": r"([#@]\w+)",
        "url": r"(?:\w+:/{2})?(?:www\.)?([a-z\d\-]+)\.(?:[a-z\d\.]{2,})(?:/[a-zA-Z/\d]*)?",
        "punctuation": r"[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff`]+",
        "apostrophe": r"'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'",
        "hifen": r'\s*-\s*',
        "more_spaces": r'\s+'
    }

    text = str(raw_text) if raw_text is not None else ""

    # Translate emojis into tokens
    if no_emojis:
        text = emoji.demojize(text, delimiters=("emoji_", ""))

    # Handle hashtags
    if no_hashtags:
        if hashtag_retain_words:
            text = re.sub(patterns["hashtags_at"], "", text)
        else:
            text = re.sub(patterns["hashtags_ats_word"], r"\1", text)

    # Replace newlines
    if no_newlines:
        text = re.sub(patterns["newline"], " ", text)

    # Normalize URLs → keep only domain entity
    if no_urls:
        text = re.sub(patterns["url"], r"\1", text)  
        # Example: "www.instagram.com" → "instagram"

    # Remove punctuation
    if no_punctuation:
        text = re.sub(patterns["punctuation"], "", text)
        text = re.sub(patterns["apostrophe"], "", text)
        text = re.sub(patterns["hifen"], " ", text)

    # Clean up extra spaces
    text = re.sub(patterns["more_spaces"], " ", text)
    
    return text.strip()

# ------------------------------------------------------------------------------
# CORRECTING WRONGLY WRITTEN WORDS
# ------------------------------------------------------------------------------

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

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def vectorize_texts(texts, vectorizer_type="tfidf", max_features=1000, ngram_range=(1, 1)):
    """
    Vectorizes text data using TF-IDF, Bag-of-Words, or Doc2Vec.
    
    Parameters:
    -----------
    texts : list or pd.Series
        The text data (can be tokenized lists or strings).
    vectorizer_type : str
        "tfidf", "count", or "doc2vec".
    max_features : int
        Maximum number of features (ignored for Doc2Vec).
    ngram_range : tuple
        The n-gram range to consider (only for TF-IDF and CountVectorizer).
    
    Returns:
    --------
    dtm : sparse matrix or numpy array
        Document-Term Matrix (TF-IDF/BoW) or document vectors (Doc2Vec).
    vectorizer : fitted vectorizer object or Doc2Vec model
    """
    
    if vectorizer_type in ["tfidf", "count"]:
        # Convert tokenized lists to strings
        processed_texts = pd.Series(texts).apply(
            lambda x: " ".join(x) if isinstance(x, (list, tuple)) else str(x)
        )
    
        if vectorizer_type == "count":
            vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        else:  # TF-IDF
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        
        dtm = vectorizer.fit_transform(processed_texts)
    
    elif vectorizer_type == "doc2vec":
        # Ensure texts are tokenized lists
        tokenized_texts = [
            x if isinstance(x, (list, tuple)) else str(x).split() for x in texts
        ]
        tagged_docs = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(tokenized_texts)]
        
        vectorizer = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)
        vectorizer.build_vocab(tagged_docs)
        vectorizer.train(tagged_docs, total_examples=vectorizer.corpus_count, epochs=vectorizer.epochs)
        
        dtm = [vectorizer.infer_vector(doc.words) for doc in tagged_docs]
    
    else:
        raise ValueError("Invalid vectorizer_type. Choose 'tfidf', 'count', or 'doc2vec'.")
    
    return dtm, vectorizer



# ------------------------------------------------------------------------------
# A CLASS WITH THE MAIN PIPELINE JUST TO TEST SOMETHINGS!!
# ------------------------------------------------------------------------------

class MainPipeline(BaseEstimator):
    def __init__(self, 
                 print_output = False, 
                 no_emojis = True, 
                 no_hashtags = True,
                 hashtag_retain_words = True,
                 no_newlines = True,
                 no_urls = True,
                 no_punctuation = True,
                 no_stopwords = True,
                 custom_stopwords = [],
                 convert_diacritics = True, 
                 lowercase = True, 
                 lemmatized = True,
                 list_pos = ["n","v","a","r","s"],
                 pos_tags_list = "no_pos",
                 tokenized_output = False):
        
        self.print_output = print_output 
        self.no_emojis = no_emojis
        self.no_hashtags = no_hashtags
        self.hashtag_retain_words = hashtag_retain_words
        self.no_newlines = no_newlines
        self.no_urls = no_urls
        self.no_punctuation = no_punctuation
        self.no_stopwords = no_stopwords
        self.custom_stopwords = custom_stopwords
        self.convert_diacritics = convert_diacritics
        self.lowercase = lowercase
        self.lemmatized = lemmatized
        self.list_pos = list_pos
        self.pos_tags_list = pos_tags_list
        self.tokenized_output = tokenized_output

    def regex_cleaner(self, raw_text):

        #patterns
        newline_pattern = "(\\n)"
        hashtags_at_pattern = "([#\@@\u0040\uFF20\uFE6B])"
        hashtags_ats_and_word_pattern = "([#@]\w+)"
        emojis_pattern = "([\u2600-\u27FF])"
        url_pattern = "(?:\w+:\/{2})?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(\/[a-zA-Z\/\d]+)?" ##Note that this URL pattern is *even better*
        punctuation_pattern = "[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u007C\u2010-\u2028\ufeff`]+"
        apostrophe_pattern = "'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'"
        separated_words_pattern = "(?<=\w\s)([A-Z]\s){2,}"
        ##note that this punctuation_pattern doesn't capture ' this time to allow our tokenizer to separate "don't" into ["do", "n't"]
        
        if self.no_emojis == True:
            clean_text = re.sub(emojis_pattern,"",raw_text)
        else:
            clean_text = raw_text

        if self.no_hashtags == True:
            if self.hashtag_retain_words == True:
                clean_text = re.sub(hashtags_at_pattern,"",clean_text)
            else:
                clean_text = re.sub(hashtags_ats_and_word_pattern,"",clean_text)
            
        if self.no_newlines == True:
            clean_text = re.sub(newline_pattern," ",clean_text)

        if self.no_urls == True:
            clean_text = re.sub(url_pattern,"",clean_text)
        
        if self.no_punctuation == True:
            clean_text = re.sub(punctuation_pattern,"",clean_text)
            clean_text = re.sub(apostrophe_pattern,"",clean_text)

        return clean_text

    def lemmatize_all(self, token):
    
        wordnet_lem = nltk.stem.WordNetLemmatizer()
        for arg_1 in self.list_pos[0]:
            token = wordnet_lem.lemmatize(token, arg_1)
        return token

    def main_pipeline(self, raw_text):
        
        """Preprocess strings according to the parameters"""
        if self.print_output == True:
            print("Preprocessing the following input: \n>> {}".format(raw_text))

        clean_text = self.regex_cleaner(raw_text) 

        if self.print_output == True:
            print("Regex cleaner returned the following: \n>> {}".format(clean_text))

        tokenized_text = nltk.tokenize.word_tokenize(clean_text)

        tokenized_text = [re.sub("'m","am",token) for token in tokenized_text]
        tokenized_text = [re.sub("n't","not",token) for token in tokenized_text]
        tokenized_text = [re.sub("'s","is",token) for token in tokenized_text]

        if self.no_stopwords == True:
            stopwords = nltk.corpus.stopwords.words("english")
            tokenized_text = [item for item in tokenized_text if item.lower() not in stopwords]
        
        if self.convert_diacritics == True:
            tokenized_text = [unidecode(token) for token in tokenized_text]

        if self.lemmatized == True:
            tokenized_text = [self.lemmatize_all(token) for token in tokenized_text]
    
        if self.no_stopwords == True:
            tokenized_text = [item for item in tokenized_text if item.lower() not in self.custom_stopwords]

        if self.pos_tags_list == "pos_list" or self.pos_tags_list == "pos_tuples" or self.pos_tags_list == "pos_dictionary":
            pos_tuples = nltk.tag.pos_tag(tokenized_text)
            pos_tags = [pos[1] for pos in pos_tuples]
        
        if self.lowercase == True:
            tokenized_text = [item.lower() for item in tokenized_text]
        
        if self.pos_tags_list == "pos_list":
            return (tokenized_text, pos_tags)
        elif self.pos_tags_list == "pos_tuples":
            return pos_tuples   
        
        else:
            if self.tokenized_output == True:
                return tokenized_text
            else:
                detokenizer = TreebankWordDetokenizer()
                detokens = detokenizer.detokenize(tokenized_text)
                if self.print_output == True:
                    print("Pipeline returning the following result: \n>> {}".format(str(detokens)))
                return str(detokens)
        
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
# TRANSLATING NON-ENGLISH-REVIEWS
#---------------------------------------------------------------------------------------------------

def process_and_translate_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Detects language of reviews, translates non-English rows to English,
    and prepares unified columns for downstream pipeline processing.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input DataFrame with a column '00_before_translating_cleaning'.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with added columns:
        - 'lang_langdetect'
        - 'lang_langid'
        - 'needs_translation'
        - 'text_translated'
        - 'text_for_pipeline'
    """

    # --- Safe language detection helpers ---
    def safe_detect_lang_langdetect(text):
        if ld_detect is None:
            return None
        try:
            return ld_detect(str(text))
        except Exception:
            return None

    def safe_detect_lang_langid(text):
        if langid is None:
            return None
        try:
            return langid.classify(str(text))[0]
        except Exception:
            return None

    # --- Run detection ---
    dataset['lang_langdetect'] = dataset['00_before_translating_cleaning'].apply(safe_detect_lang_langdetect)
    dataset['lang_langid'] = dataset['00_before_translating_cleaning'].apply(safe_detect_lang_langid)

    # --- Decide if translation is needed ---
    def need_translation_row(row):
        ld = row['lang_langdetect']
        li = row['lang_langid']
        if ld is None and li is None:
            return True
        if ld is not None and ld != 'en':
            return True
        if li is not None and li != 'en':
            return True
        return False

    dataset['needs_translation'] = dataset.apply(need_translation_row, axis=1)

    # --- Translator setup ---
    translator = None
    if GoogleTranslator is not None:
        translator = GoogleTranslator(source='auto', target='en')

    def translate_safe(text):
        if translator is None:
            return str(text)
        try:
            return translator.translate(str(text))
        except Exception:
            return str(text)

    # --- Translate only where needed ---
    dataset['text_translated'] = dataset.apply(
        lambda r: translate_safe(r['00_before_translating_cleaning']) if r['needs_translation'] else r['00_before_translating_cleaning'],
        axis=1
    )

    # --- Unified column for downstream pipeline ---
    dataset['text_for_pipeline'] = dataset['text_translated']

    return dataset

#---------------------------------------------------------------------------------------------------
# FEATURE EXTRACTION FOR NER
#---------------------------------------------------------------------------------------------------

def word2features(token_list, POS_list, i):
    word = token_list[i]
    postag = POS_list[i]

    features = {
        'bias': 1.0, #prior probability of the label
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    
    if i > 0:
        word1 = token_list[i-1]
        postag1 = POS_list[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(token_list)-1:
        word1 = token_list[i+1]
        postag1 = POS_list[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(token_list, POS_list):
    return [word2features(token_list, POS_list, i) for i in range(len(token_list))]

# ------------------------------------------------------------------------------
# CREATION OF BIO NER TAGS
# ------------------------------------------------------------------------------

def align_bio(doc, tokens):
    """
    Convert spaCy entity spans into BIO tags aligned to a list of tokens.
    """
    bio_labels = ["O"] * len(tokens)

    for ent in doc.ents:
        ent_tokens = [t.text for t in ent]

        # Find where the entity starts in our token list
        for i in range(len(tokens)):
            # Match entity beginning
            if tokens[i:i+len(ent_tokens)] == ent_tokens:
                bio_labels[i] = f"B-{ent.label_}"
                # Mark inside tokens
                for j in range(i+1, i+len(ent_tokens)):
                    bio_labels[j] = f"I-{ent.label_}"
                break

    return bio_labels

def align_bio_to_custom_tokens(text, tokens, nlp, equivalence_table):
    doc = nlp(text)
    bio = ["O"] * len(tokens)

    for ent in doc.ents:
        ent_tokens = [t.text for t in ent]

        for i in range(len(tokens)):
            if tokens[i:i+len(ent_tokens)] == ent_tokens:
                bio[i] = "B" + equivalence_table[ent.label_]
                for j in range(i+1, i+len(ent_tokens)):
                    bio[j] = "I" + equivalence_table[ent.label_]
                break

    return bio

