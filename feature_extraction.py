import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


_SENTIMENT_CRITICAL = {
    "not",
    "no",
    "nor",
    "never",
    "nothing",
    "nobody",
    "nowhere",
    "neither",
    "very",
    "too",
    "so",
    "more",
    "most",
    "less",
    "least",
    "quite",
    "rather",
    "really",
    "truly",
    "just",
    "only",
}

_base_stopwords = set(stopwords.words("english"))


STOP_WORDS = _base_stopwords - _SENTIMENT_CRITICAL

STEMMER = PorterStemmer()


def clean_text(text):
    """
    Standardizes text by lowercasing, removing punctuation,
    removing stop words, and stemming the core words.
    Sentiment-critical words (not, very, never, no, etc.) are intentionally
    kept so the model can distinguish "good" from "not good".
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    cleaned_words = [STEMMER.stem(w) for w in words if w not in STOP_WORDS]
    return cleaned_words


def l2_normalize(matrix):
    """
    L2-normalize each row so that document length doesn't dominate the
    feature magnitude. Short and long reviews become comparable.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return matrix / norms


class BagOfWord:
    def __init__(self, min_freq=3):
        self.vocab = {}
        self.min_freq = min_freq

    def fit(self, sent_list):
        word_counts = {}
        for sent in sent_list:
            for word in clean_text(sent):
                word_counts[word] = word_counts.get(word, 0) + 1
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.vocab[word] = len(self.vocab)

    def transform(self, sent_list, normalize=True):
        feat = np.zeros((len(sent_list), len(self.vocab)), dtype=np.float32)
        for idx, sent in enumerate(sent_list):
            for word in clean_text(sent):
                if word in self.vocab:
                    feat[idx][self.vocab[word]] += 1
        return l2_normalize(feat) if normalize else feat

    def fit_transform(self, sent_list, normalize=True):
        self.fit(sent_list)
        return self.transform(sent_list, normalize)


class TFIDF:
    def __init__(self, min_freq=3):
        self.vocab = {}
        self.idf = {}
        self.min_freq = min_freq

    def fit(self, sent_list):
        word_counts = {}
        for sent in sent_list:
            for word in clean_text(sent):
                word_counts[word] = word_counts.get(word, 0) + 1
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.vocab[word] = len(self.vocab)

        N = len(sent_list)
        doc_freq = {word: 0 for word in self.vocab}
        for sent in sent_list:
            for word in set(clean_text(sent)):
                if word in self.vocab:
                    doc_freq[word] += 1

        self.idf = {word: np.log(N / (df + 1)) for word, df in doc_freq.items()}

    def transform(self, sent_list, normalize=True):
        feat = np.zeros((len(sent_list), len(self.vocab)), dtype=np.float32)
        for idx, sent in enumerate(sent_list):
            words = clean_text(sent)
            doc_len = max(len(words), 1)
            doc_tf = {}
            for word in words:
                if word in self.vocab:
                    doc_tf[word] = doc_tf.get(word, 0) + 1
            for word, count in doc_tf.items():
                feat[idx][self.vocab[word]] = (count / doc_len) * self.idf[word]
        return l2_normalize(feat) if normalize else feat

    def fit_transform(self, sent_list, normalize=True):
        self.fit(sent_list)
        return self.transform(sent_list, normalize)


class NGram:
    def __init__(self, ngram, min_freq=3, binary=False):
        self.ngram = ngram
        self.feature_map = {}
        self.min_freq = min_freq
        self.binary = binary

    def fit(self, sent_list):
        gram_counts = {}
        for gram_n in self.ngram:
            for sent in sent_list:
                words = clean_text(sent)
                for i in range(len(words) - gram_n + 1):
                    feature = "_".join(words[i : i + gram_n])
                    gram_counts[feature] = gram_counts.get(feature, 0) + 1
        for feature, count in gram_counts.items():
            if count >= self.min_freq:
                self.feature_map[feature] = len(self.feature_map)

    def transform(self, sent_list, normalize=True):
        feat = np.zeros((len(sent_list), len(self.feature_map)), dtype=np.float32)
        for idx, sent in enumerate(sent_list):
            words = clean_text(sent)
            for gram_n in self.ngram:
                for i in range(len(words) - gram_n + 1):
                    feature = "_".join(words[i : i + gram_n])
                    if feature in self.feature_map:
                        if self.binary:
                            feat[idx][self.feature_map[feature]] = 1
                        else:
                            feat[idx][self.feature_map[feature]] += 1
        return l2_normalize(feat) if normalize else feat

    def fit_transform(self, sent_list, normalize=True):
        self.fit(sent_list)
        return self.transform(sent_list, normalize)
