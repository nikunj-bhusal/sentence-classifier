import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# - re: Regular expressions: a powerful tool for pattern matching in text. Here it's used to remove punctuation (keep only letters and spaces).

# - nltk: Natural Language Toolkit. A library with many text processing tools (stop words, stemmers, tokenizers, etc.).

# - stopwords: A specific NLTK module that provides a list of common "filler" words like "the", "and", "is", words that usually don't carry much meaning.

# - PorterStemmer: An algorithm that reduces words to their root form (e.g., "running" → "run", "better" → "better": Porter stemmer is rule‑based).


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# What it does: Checks if the stopwords dataset is already downloaded in your NLTK data folder. If not, it downloads it quietly (without printing progress).

# Why it's needed: NLTK requires you to download corpora (data) separately. Without this, trying to use stopwords.words("english") would crash.


SENTIMENT_CRITICAL_WORDS = {
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
# What it is: A Python set of words that are important for detecting sentiment.

# Why these words are kept: In typical text preprocessing, we remove common stop words (like "the", "a", "and") because they are not informative. However, words like "not" or "never" can flip the sentiment (e.g., "not good" vs "good"). Words like "very" or "really" act as intensity modifiers. If we removed them, the model would lose crucial clues. So we will keep them even though they appear in the standard stopword list.

module_stopwords = set(stopwords.words("english"))
STOP_WORDS = module_stopwords - SENTIMENT_CRITICAL_WORDS
# module_stopwords : The full set of standard English stopwords from NLTK (around 179 words like "i", "you", "he", "she", "it", "we", "they", "a", "an", "the", "and", "of", "to", etc.)

# STOP_WORDS : Words to be removed later. We subtract the two to keep all ordinary stopwords except the sentiment‑critical ones, so that the critical stopwords will not be filtered out later.

STEMMER = PorterStemmer()
# What it does: Creates an instance of the Porter Stemmer.

# Why it's used: Stemming reduces words to a common base form so that different grammatical forms are treated the same (e.g., "love", "loves", "loving" → all become "love"). This reduces the number of unique words (vocabulary size) and helps the model generalise.


def clean_text(text):
    # Standardizes text by:
    # - lowercasing,
    # - removing punctuation,
    # - removing stop words, and
    # - stemming the core words.
    # Sentiment-critical words (not, very, never, no, etc.) are intentionally
    # kept so the model can distinguish "good" from "not good".
    text = text.lower()

    text = re.sub(r"[^a-z\s]", "", text)
    # re.sub() replaces all characters that are not (^) lowercase letters (a-z) or whitespace (\s) with an empty string (i.e., removes them).

    words = text.split()
    # Splits the cleaned string on whitespace into a list of words.
    # Why: We need individual tokens (words) to process further.

    cleaned_words = [STEMMER.stem(word) for word in words if word not in STOP_WORDS]
    # Why stem: Stemming collapses word variants ("loves", "loving", "loved" → "love"). This makes the feature space smaller and helps generalisation.
    return cleaned_words


def l2_normalize(matrix):
    # This function takes a 2D NumPy array (matrix) where each row represents one document (e.g., a review) and each column is a feature (e.g., word count). It returns the same matrix with rows normalised to have Euclidean (L2) length = 1.
    # Why?: so that the document length (total word count) does not bias the model, allowing for focus on relative word frequencies.

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # `np.linalg.norm` computes the Euclidean length of each row.
    # `axis=1` means along the columns (for each row, compute one norm).
    # `keepdims=True` keeps the result as a column vector: shape (n_rows, 1) instead of (n_rows,).
    # Why: We want to divide each row by its own length, so we need the lengths in a shape that can broadcast with the matrix.

    norms = np.where(norms == 0, 1, norms)
    # If a row is all zeros (e.g., an empty sentence after cleaning), its norm would be 0. Dividing by 0 is invalid.
    # This line replaces any zero norm with 1. That way, dividing by norms leaves that row unchanged (still all zeros) instead of crashing.

    # After division, each row becomes a unit vector.
    return matrix / norms


# TF (Term Frequency): How often a word appears in a specific document (review).
# Eg: Common words like "the" appear often but aren't informative.

# IDF (Inverse Document Frequency): How rare a word is across all documents.
# Rare words get higher IDF, common words get lower IDF.

# TF‑IDF = TF × IDF.
# It gives high weight to words that are frequent in one document but rare overall. these tend to be the most meaningful words for that document


class TFIDF:
    def __init__(self, min_freq=3):
        self.vocab = {}  # maps column index
        self.idf = {}  # maps word -> its IDF score
        self.min_freq = 3  # ignore words that appear fewer than 3 times in total

    def fit(self, sentence_list):
        word_counts = {}
        for sentence in sentence_list:
            for word in clean_text(sentence):  # clean each sentence then
                # count number of occurrences of each word in the sentence
                word_counts[word] = word_counts.get(word, 0) + 1

        # for each word in a sentence,
        for word, count in word_counts.items():
            if count >= self.min_freq:  # if it appears more than 3 times.
                # save the word in list of vocab with "number of words known in vocab" as value
                self.vocab[word] = len(self.vocab)

        N = len(sentence_list)  # total number of sentences received in argument

        # createing a dictionary initialising the document frequency to 0 for every word already in the vocabulary.
        doc_freq = {word: 0 for word in self.vocab}
        for sentence in sentence_list:
            for word in set(clean_text(sentence)):  # for each word in cleaned sentence,
                if word in self.vocab:
                    # if the word occurred >3 times and got saved to vocab, increase its count
                    doc_freq[word] += 1

        # calculare idf score: IDF(term) = log(N / no_of_documents_containing(term))
        # for each word in doc_freq, calculate idf score
        # + 1 to avoid division by zero error
        self.idf = {word: np.log(N / (df + 1)) for word, df in doc_freq.items()}

    def transform(self, sentence_list, normalize=True):
        feat = np.zeros((len(sentence_list), len(self.vocab)), dtype=np.float32)
        for idx, sentence in enumerate(sentence_list):
            words = clean_text(sentence)
            doc_len = max(len(words), 1)
            doc_tf = {}
            for word in words:
                if word in self.vocab:
                    doc_tf[word] = doc_tf.get(word, 0) + 1
            for word, count in doc_tf.items():
                feat[idx][self.vocab[word]] = (count / doc_len) * self.idf[word]
        return l2_normalize(feat) if normalize else feat

    def fit_transform(self, sentence_list, normalize=True):
        self.fit(sentence_list)
        return self.transform(sentence_list, normalize)


class NGram:
    def __init__(self, ngram, min_freq=3, binary=False):
        self.ngram = ngram
        self.feature_map = {}
        self.min_freq = min_freq
        self.binary = binary

    def fit(self, sentence_list):
        gram_counts = {}
        for gram_n in self.ngram:
            for sentence in sentence_list:
                words = clean_text(sentence)
                for i in range(len(words) - gram_n + 1):
                    feature = "_".join(words[i : i + gram_n])
                    gram_counts[feature] = gram_counts.get(feature, 0) + 1
        for feature, count in gram_counts.items():
            if count >= self.min_freq:
                self.feature_map[feature] = len(self.feature_map)

    def transform(self, sentence_list, normalize=True):
        feat = np.zeros((len(sentence_list), len(self.feature_map)), dtype=np.float32)
        for idx, sentence in enumerate(sentence_list):
            words = clean_text(sentence)
            for gram_n in self.ngram:
                for i in range(len(words) - gram_n + 1):
                    feature = "_".join(words[i : i + gram_n])
                    if feature in self.feature_map:
                        if self.binary:
                            feat[idx][self.feature_map[feature]] = 1
                        else:
                            feat[idx][self.feature_map[feature]] += 1
        return l2_normalize(feat) if normalize else feat

    def fit_transform(self, sentence_list, normalize=True):
        self.fit(sentence_list)
        return self.transform(sentence_list, normalize)
