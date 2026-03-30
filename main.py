import numpy as np
from data_preprocess import read_data
from feature_extraction import BagOfWord, NGram
from softmax_regression import SoftmaxRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    debug = 1  # Turn debug ON
    X_data, y_data = read_data()

    if debug == 1:
        # Limit data to 10,000 samples to fit into standard computer memory
        X_data = X_data[:50000]
        y_data = y_data[:50000]

    y = np.array(y_data).reshape(len(y_data), 1)

    # 1. Feature Extraction (UPDATED: Removed do_lower_case arguments!)
    bag_of_word_model = BagOfWord()
    ngram_model = NGram(ngram=(1,))

    print(
        "Extracting features... this might take a few seconds with the new text cleaner!"
    )
    X_Bow = bag_of_word_model.fit_transform(X_data)
    X_Gram = ngram_model.fit_transform(X_data)

    print("BoW shape", X_Bow.shape)
    print("Gram shape", X_Gram.shape)

    # 2. Split into Train and Test sets
    X_train_Bow, X_test_Bow, y_train_Bow, y_test_Bow = train_test_split(
        X_Bow, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_Gram, X_test_Gram, y_train_Gram, y_test_Gram = train_test_split(
        X_Gram, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Setup Training Parameters
    epoch = 50  # Reduced epochs because we have 5x more data
    bow_learning_rate = 0.1  # Lowered from 1 to 0.1 for stability
    gram_learning_rate = 0.1  # Lowered from 1 to 0.1 for stability

    # 4. Train Model 1 (Bag of Words)
    print("--- Training BoW Model ---")
    model1 = SoftmaxRegression()
    history1 = model1.fit(
        X_train_Bow,
        y_train_Bow,
        epoch=epoch,
        learning_rate=bow_learning_rate,
        print_loss_steps=epoch // 10,
        update_strategy="stochastic",
    )
    plt.plot(np.arange(len(history1)), np.array(history1))
    plt.title("BoW Model Loss over Time")
    plt.show()
    print(
        "BoW train accuracy: {} | test accuracy: {}".format(
            model1.score(X_train_Bow, y_train_Bow), model1.score(X_test_Bow, y_test_Bow)
        )
    )

    # 5. Train Model 2 (N-Gram)
    print("--- Training N-Gram Model ---")
    model2 = SoftmaxRegression()
    history2 = model2.fit(
        X_train_Gram,
        y_train_Gram,
        epoch=epoch,
        learning_rate=gram_learning_rate,
        print_loss_steps=epoch // 10,
        update_strategy="stochastic",
    )
    plt.plot(np.arange(len(history2)), np.array(history2))
    plt.title("N-Gram Model Loss over Time")
    plt.show()
    print(
        "Gram train accuracy: {} | test accuracy: {}".format(
            model2.score(X_train_Gram, y_train_Gram),
            model2.score(X_test_Gram, y_test_Gram),
        )
    )

    # 6. Real-time Inference Loop
    print("\n" + "=" * 30)
    print("AI SENTIMENT PREDICTOR")
    print("=" * 30)

    # Map the numerical outputs back to human-readable strings
    sentiment_map = {
        0: "Very Negative 😠",
        1: "Negative 🙁",
        2: "Neutral 😐",
        3: "Positive 🙂",
        4: "Very Positive 😍",
    }

    while True:
        user_input = input("\nEnter a movie review (or type 'exit' to quit): ")

        if user_input.lower() == "exit":
            break

        # Put the single sentence into a list so the transform method can process it
        test_phrase = [user_input]

        # Step A: Transform the text into numerical features
        bow_features = bag_of_word_model.transform(test_phrase)
        gram_features = ngram_model.transform(test_phrase)

        # Step B: Get predictions from both models
        bow_prediction = model1.predict(bow_features)[0]
        gram_prediction = model2.predict(gram_features)[0]

        # Step C: Display results
        print(f"-> BoW Model Prediction:    {sentiment_map[bow_prediction]}")
        print(f"-> N-Gram Model Prediction: {sentiment_map[gram_prediction]}")
