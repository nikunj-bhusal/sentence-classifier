import numpy as np
from data_preprocess import read_data
from feature_extraction import TFIDF, NGram
from softmax_regression import SoftmaxRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

if __name__ == "__main__":

    debug = 1
    X_data, y_data = read_data()

    if debug == 1:
        X_data = X_data[:50000]
        y_data = y_data[:50000]

    y = np.array(y_data).flatten().astype(int)

    y_mapped = np.zeros_like(y)
    y_mapped[(y == 0) | (y == 1)] = 0
    y_mapped[y == 2] = 1
    y_mapped[(y == 3) | (y == 4)] = 2
    y = y_mapped

    print("Dataset label distribution:")
    names = ["Negative", "Neutral ", "Positive"]
    for c in range(3):
        count = int(np.sum(y == c))
        bar = "█" * int(count / len(y) * 40)
        print(f"  {names[c]} ({c}): {count:6d} ({count/len(y)*100:4.1f}%)  {bar}")
    print()

    tfidf_model = TFIDF(min_freq=5)
    ngram_model = NGram(ngram=(1, 2), min_freq=5, binary=False)

    print("Extracting TF-IDF features...")
    X_Tfidf_dense = tfidf_model.fit_transform(X_data, normalize=True)
    X_Tfidf = csr_matrix(X_Tfidf_dense)
    del X_Tfidf_dense
    print(
        f"  TF-IDF shape: {X_Tfidf.shape}  "
        f"density: {X_Tfidf.nnz / (X_Tfidf.shape[0]*X_Tfidf.shape[1])*100:.2f}%  "
        f"memory: ~{X_Tfidf.data.nbytes/1e6:.0f} MB"
    )

    print("Extracting N-Gram features...")
    X_Gram_dense = ngram_model.fit_transform(X_data, normalize=True)
    X_Gram = csr_matrix(X_Gram_dense)
    del X_Gram_dense
    print(
        f"  N-Gram  shape: {X_Gram.shape}  "
        f"density: {X_Gram.nnz / (X_Gram.shape[0]*X_Gram.shape[1])*100:.2f}%  "
        f"memory: ~{X_Gram.data.nbytes/1e6:.0f} MB"
    )

    X_tr_tfidf, X_te_tfidf, y_tr, y_te = train_test_split(
        X_Tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr_gram, X_te_gram, y_tr_g, y_te_g = train_test_split(
        X_Gram, y, test_size=0.2, random_state=42, stratify=y
    )

    del X_Tfidf, X_Gram

    epoch = 40
    learning_rate = 0.3
    lambda_reg = 0.0005
    lr_decay = 0.98
    batch_size = 256

    print("\n--- Training TF-IDF Model ---")
    model1 = SoftmaxRegression()
    history1 = model1.fit(
        X_tr_tfidf,
        y_tr,
        num_of_class=3,
        epoch=epoch,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
        update_strategy="mini_batch",
        batch_size=batch_size,
        lr_decay=lr_decay,
        print_loss_steps=5,
        use_class_weights=True,
    )

    plt.figure()
    plt.plot(history1)
    plt.title("TF-IDF Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()

    print(
        f"TF-IDF  train: {model1.score(X_tr_tfidf, y_tr):.4f} | "
        f"test:  {model1.score(X_te_tfidf, y_te):.4f}"
    )
    preds1 = model1.predict(X_te_tfidf)
    print("  Per-class accuracy:")
    for c in range(3):
        mask = y_te == c
        if mask.sum() > 0:
            acc = np.mean(preds1[mask] == c)
            print(f"    {names[c]}: {acc:.3f}  (n={mask.sum()})")

    print("\n--- Training N-Gram Model ---")
    model2 = SoftmaxRegression()
    history2 = model2.fit(
        X_tr_gram,
        y_tr_g,
        num_of_class=3,
        epoch=epoch,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
        update_strategy="mini_batch",
        batch_size=batch_size,
        lr_decay=lr_decay,
        print_loss_steps=5,
        use_class_weights=True,
    )

    plt.figure()
    plt.plot(history2)
    plt.title("N-Gram Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()

    print(
        f"N-Gram  train: {model2.score(X_tr_gram, y_tr_g):.4f} | "
        f"test:  {model2.score(X_te_gram, y_te_g):.4f}"
    )
    preds2 = model2.predict(X_te_gram)
    print("  Per-class accuracy:")
    for c in range(3):
        mask = y_te_g == c
        if mask.sum() > 0:
            acc = np.mean(preds2[mask] == c)
            print(f"    {names[c]}: {acc:.3f}  (n={mask.sum()})")

    # 6. Inference
    print("\n" + "=" * 40)
    print("AI SENTIMENT PREDICTOR")
    print("=" * 40)

    sentiment_map = {
        0: "Negative 🙁",
        1: "Neutral 😐",
        2: "Positive 🙂",
    }

    while True:
        user_input = input("\nEnter a movie review (or 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        phrase = [user_input]
        # transform() returns dense; convert to sparse for the model
        tfidf_feat = csr_matrix(tfidf_model.transform(phrase, normalize=True))
        gram_feat = csr_matrix(ngram_model.transform(phrase, normalize=True))

        p1 = model1.predict(tfidf_feat)[0]
        p2 = model2.predict(gram_feat)[0]
        proba1 = model1.predict_proba(tfidf_feat)[0]
        proba2 = model2.predict_proba(gram_feat)[0]

        bar1 = "  ".join(f"{names[i].strip()}:{proba1[i]*100:.0f}%" for i in range(3))
        bar2 = "  ".join(f"{names[i].strip()}:{proba2[i]*100:.0f}%" for i in range(3))
        print(f"\n  TF-IDF -> {sentiment_map[p1]:22s}  conf:{proba1[p1]*100:.1f}%")
        print(f"    {bar1}")
        print(f"  N-Gram -> {sentiment_map[p2]:22s}  conf:{proba2[p2]*100:.1f}%")
        print(f"    {bar2}")
