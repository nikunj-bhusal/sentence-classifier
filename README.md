# Sentence Classification


This project is a **Sentence Classifier** designed to read sentences and predict their sentiment (from *0 = Very Negative* to *4 = Very Positive*). Instead of relying on heavy machine learning libraries like TensorFlow or Scikit-Learn, this project implements the core algorithms entirely from scratch using raw Python and matrix math (`NumPy`).

## Project Objectives
1. **Understand Text Processing:** To learn how computers convert human language (text) into numbers that a mathematical model can process.
2. **Build the Engine:** To write the actual training loop and optimization algorithms (Gradient Descent) manually, rather than just importing a pre-built tool.

---

## Theoretical Knowledge Required

Things to know to understand this project:
**How we turn words into math**, and **How the computer makes a guess**.

### 1. Feature Extraction (Text Vectorization)

Because machine learning algorithms operate strictly on numerical data, raw text must be transformed into quantitative arrays. This project utilizes two primary techniques for this process:

* **Bag of Words (BoW):** This method disregards grammar and word sequence, treating a sentence simply as a collection (or "bag") of its individual words. It focuses entirely on word frequency. For instance, if the word "terrible" appears frequently in a phrase, the model receives a strong numerical indicator of negative sentiment.
* **N-Grams:** Because the BoW model ignores sequence, it misses context (e.g., failing to distinguish between "good" and "not good"). N-Grams solve this by grouping adjacent words into sequences of length *n*. For example, a 2-gram (Bigram) of the phrase "I love coding" generates the distinct features "I_love" and "love_coding," preserving local word order and providing essential context to the model.

### 2. Softmax Regression (Predictive Model)

After extracting numerical features from the text, the data is passed into our classification model: **Softmax Regression** (also known as Multinomial Logistic Regression).

* **The Weights Matrix:** The model maintains a mathematical matrix of learned parameters called "weights." Each feature (whether a single word or an N-gram) has a specific weight associated with each of the sentiment categories. During training, the model learns to mathematically increase the weight of positive words for the "Positive" category while decreasing their weight for the "Negative" category.
* **The Softmax Function:** Initially, the model multiplies the input features by the weights to generate raw, unnormalized scores (called logits) for each of the five sentiment classes. The Softmax function is then applied to mathematically normalize these arbitrary scores into a standardized probability distribution.

The outputs are transformed into percentages that sum exactly to 1 (or 100%), representing the model's confidence for each class (e.g., 85% Positive, 10% Neutral, 5% Negative).


$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

### 3. Gradient Descent (Optimization Algorithm)

During the training phase, the model's predictions are continuously compared against the actual target labels. The discrepancy between the predicted probability and the true label is quantified using a **Loss Function** (specifically, Cross-Entropy Loss).

To minimize this error, the model uses **Gradient Descent optimization algorithm**. This algorithm calculates the mathematical gradient of the loss function and iteratively updates the model's weight matrix in the opposite direction of the gradient. By repeating this process over many iterations (epochs), the model gradually reduces the loss and improves its predictive accuracy.

---

## How to Run

**1. Install Dependencies**
Install basic data science libraries:
```bash
# scikit-learn for train_test_split
pip install numpy pandas matplotlib scikit-learn
```

**2. Run the Code**
```bash
python main.py
```
