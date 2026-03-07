import numpy as np
from data_preprocess import read_data
from feature_extraction import BagOfWord, NGram
from softmax_regression import SoftmaxRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    debug = 0
    X_data, y_data = read_data()

    if debug == 1:
        # Limit data to 1000 samples for faster testing while building
        X_data = X_data[:1000]
        y_data = y_data[:1000]

    y = np.array(y_data).reshape(len(y_data), 1)

    # 1. Feature Extraction
    bag_of_word_model = BagOfWord(do_lower_case=True)
    ngram_model = NGram(ngram=(1, 2), do_lower_case=True)
    X_Bow = bag_of_word_model.fit_transform(X_data)
    X_Gram = ngram_model.fit_transform(X_data)

    print("BoW shape", X_Bow.shape)
    print("Gram shape", X_Gram.shape)

    # 2. Split into Train and Test sets
    X_train_Bow, X_test_Bow, y_train_Bow, y_test_Bow = train_test_split(X_Bow, y, test_size=0.2, random_state=42, stratify=y)
    X_train_Gram, X_test_Gram, y_train_Gram, y_test_Gram = train_test_split(X_Gram, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Setup Training Parameters
    epoch = 100
    bow_learning_rate = 1
    gram_learning_rate = 1

    # 4. Train Model 1 (Bag of Words)
    print("--- Training BoW Model ---")
    model1 = SoftmaxRegression()
    history1 = model1.fit(X_train_Bow, y_train_Bow, epoch=epoch, learning_rate=bow_learning_rate, print_loss_steps=epoch//10, update_strategy="stochastic")
    plt.plot(np.arange(len(history1)), np.array(history1))
    plt.title("BoW Model Loss over Time")
    plt.show()
    print("BoW train accuracy: {} | test accuracy: {}".format(model1.score(X_train_Bow, y_train_Bow), model1.score(X_test_Bow, y_test_Bow)))

    # 5. Train Model 2 (N-Gram)
    print("--- Training N-Gram Model ---")
    model2 = SoftmaxRegression()
    history2 = model2.fit(X_train_Gram, y_train_Gram, epoch=epoch, learning_rate=gram_learning_rate, print_loss_steps=epoch//10, update_strategy="stochastic")
    plt.plot(np.arange(len(history2)), np.array(history2))
    plt.title("N-Gram Model Loss over Time")
    plt.show()
    print("Gram train accuracy: {} | test accuracy: {}".format(model2.score(X_train_Gram, y_train_Gram), model2.score(X_test_Gram, y_test_Gram)))