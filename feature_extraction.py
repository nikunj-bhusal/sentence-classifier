import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure the stopwords are downloaded on your machine
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize these once globally so the program doesn't waste time reloading them for every sentence

STOP_WORDS = set(stopwords.words('english')).union(set('ok'))
STEMMER = PorterStemmer()

def clean_text(text):
    """
    Standardizes text by lowercasing, removing punctuation,
    removing stop words, and stemming the core words.
    """
    # 1. Lowercase the text
    text = text.lower()

    # 2. Remove special characters (keep only a-z and spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Tokenize (split into a list of words)
    words = text.split()

    # 4. Remove Stop Words and Stem the remaining words
    cleaned_words = [STEMMER.stem(w) for w in words if w not in STOP_WORDS]

    return cleaned_words


class BagOfWord:
    def __init__(self):
        self.vocab = {}

    def fit(self, sent_list):
        for sent in sent_list:
            # We use our new text cleaner here!
            words = clean_text(sent)
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

    def transform(self, sent_list):
        vocab_size = len(self.vocab)
        # Using int8 to save memory!
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
    def __init__(self, ngram):
        self.ngram = ngram
        self.feature_map = {}

    def fit(self, sentList):
        for gram in self.ngram:
            for sent in sentList:
                words = clean_text(sent)
                # Create N-grams from the cleaned word list
                for i in range(len(words) - gram + 1):
                    feature = "_".join(words[i:i + gram])
                    if feature not in self.feature_map:
                        self.feature_map[feature] = len(self.feature_map)

    def transform(self, sentList):
        n = len(sentList)
        m = len(self.feature_map)
        # Using int8 to save memory!
        ngram_feature = np.zeros((n, m), dtype=np.int8)

        for idx, sent in enumerate(sentList):
            words = clean_text(sent)
            for gram in self.ngram:
                for i in range(len(words) - gram + 1):
                    feature = "_".join(words[i:i + gram])
                    if feature in self.feature_map:
                        ngram_feature[idx][self.feature_map[feature]] = 1

        return ngram_feature

    def fit_transform(self, sentList):
        self.fit(sentList)
        return self.transform(sentList)