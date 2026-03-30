import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure the stopwords are downloaded on your machine
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Initialize these once globally so the program doesn't waste time reloading them for every sentence

STOP_WORDS = set(stopwords.words("english")).union(set("ok"))
STEMMER = PorterStemmer()


def clean_text(text):
    """
    Standardizes text by lowercasing, removing punctuation,
    removing stop words, and stemming the core words.
    """
    # 1. Lowercase the text
    text = text.lower()

    # 2. Remove special characters (keep only a-z and spaces)
    text = re.sub(r"[^a-z\s]", "", text)

    # 3. Tokenize (split into a list of words)
    words = text.split()

    # 4. Remove Stop Words and Stem the remaining words
    cleaned_words = [STEMMER.stem(w) for w in words if w not in STOP_WORDS]

    return cleaned_words


class BagOfWord:
    # Added min_freq parameter (default is 3)
    def __init__(self, min_freq=3):
        self.vocab = {}
        self.min_freq = min_freq

    def fit(self, sent_list):
        # Pass 1: Count frequencies of all words
        word_counts = {}
        for sent in sent_list:
            words = clean_text(sent)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Pass 2: Only add words to vocab if they appear >= min_freq times
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.vocab[word] = len(self.vocab)

    def transform(self, sent_list):
        vocab_size = len(self.vocab)
        bag_of_word_feature = np.zeros((len(sent_list), vocab_size), dtype=np.int8)

        for idx, sent in enumerate(sent_list):
            words = clean_text(sent)
            for word in words:
                if word in self.vocab:
                    bag_of_word_feature[idx][self.vocab[word]] += 1

        return bag_of_word_feature

    def fit_transform(self, sent_list):
        self.fit(sent_list)
        return self.transform(sent_list)


class NGram:
    # Added min_freq parameter
    def __init__(self, ngram, min_freq=3):
        self.ngram = ngram
        self.feature_map = {}
        self.min_freq = min_freq

    def fit(self, sentList):
        # Pass 1: Count N-gram frequencies
        gram_counts = {}
        for gram in self.ngram:
            for sent in sentList:
                words = clean_text(sent)
                for i in range(len(words) - gram + 1):
                    feature = "_".join(words[i : i + gram])
                    gram_counts[feature] = gram_counts.get(feature, 0) + 1

        # Pass 2: Filter by min_freq
        for feature, count in gram_counts.items():
            if count >= self.min_freq:
                self.feature_map[feature] = len(self.feature_map)

    def transform(self, sentList):
        n = len(sentList)
        m = len(self.feature_map)
        ngram_feature = np.zeros((n, m), dtype=np.int8)

        for idx, sent in enumerate(sentList):
            words = clean_text(sent)
            for gram in self.ngram:
                for i in range(len(words) - gram + 1):
                    feature = "_".join(words[i : i + gram])
                    if feature in self.feature_map:
                        ngram_feature[idx][self.feature_map[feature]] = 1

        return ngram_feature

    def fit_transform(self, sentList):
        self.fit(sentList)
        return self.transform(sentList)
