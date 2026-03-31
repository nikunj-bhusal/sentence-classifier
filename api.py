import sys
import os
import threading

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(__file__))

from data_preprocess import read_data
from feature_extraction import TFIDF, NGram, clean_text
from softmax_regression import SoftmaxRegression

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {
    "status": "idle",
    "progress": 0,
    "loss_history": [],
    "log": [],
    "model_tfidf": None,
    "model_ngram": None,
    "tfidf": None,
    "ngram": None,
    "train_acc": None,
    "test_acc": None,
    "error": None,
}

NAMES = ["Negative", "Neutral", "Positive"]


def run_training(n_samples: int, epoch: int, lr: float):
    from sklearn.model_selection import train_test_split

    try:
        state["status"] = "training"
        state["loss_history"] = []
        state["log"] = []
        state["progress"] = 0

        def log(msg):
            state["log"].append(msg)

        log("Loading dataset...")
        X_data, y_data = read_data(
            os.path.join(os.path.dirname(__file__), "dataset/train.tsv")
        )
        X_data = X_data[:n_samples]
        y_data = y_data[:n_samples]

        y = np.array(y_data).flatten().astype(int)
        y_mapped = np.zeros_like(y)
        y_mapped[(y == 0) | (y == 1)] = 0
        y_mapped[y == 2] = 1
        y_mapped[(y == 3) | (y == 4)] = 2
        y = y_mapped

        for c in range(3):
            count = int(np.sum(y == c))
            log(f"  {NAMES[c]}: {count} ({count / len(y) * 100:.1f}%)")

        state["progress"] = 5

        log("Extracting TF-IDF features...")
        tfidf_model = TFIDF(min_freq=5)
        X_Tfidf = csr_matrix(tfidf_model.fit_transform(X_data, normalize=True))
        log(
            f"  Shape: {X_Tfidf.shape}, density: {X_Tfidf.nnz / (X_Tfidf.shape[0] * X_Tfidf.shape[1]) * 100:.2f}%"
        )
        state["progress"] = 15

        log("Extracting N-Gram features...")
        ngram_model = NGram(ngram=(1, 2), min_freq=5, binary=False)
        X_Gram = csr_matrix(ngram_model.fit_transform(X_data, normalize=True))
        log(
            f"  Shape: {X_Gram.shape}, density: {X_Gram.nnz / (X_Gram.shape[0] * X_Gram.shape[1]) * 100:.2f}%"
        )
        state["progress"] = 25

        X_tr_tfidf, X_te_tfidf, y_tr, y_te = train_test_split(
            X_Tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        X_tr_gram, X_te_gram, y_tr_g, y_te_g = train_test_split(
            X_Gram, y, test_size=0.2, random_state=42, stratify=y
        )

        tfidf_loss = []
        ngram_loss = []

        def patched_fit(model, X, y_arr, loss_list, label):
            from softmax_regression import (
                _add_bias,
                _dot,
                _row_slice,
                softmax,
                compute_class_weights,
            )
            from scipy.sparse import issparse

            X_b = _add_bias(X)
            n, m = X_b.shape[0], X_b.shape[1]
            y_flat = np.array(y_arr).flatten().astype(int)
            num_of_class = 3

            class_weights = compute_class_weights(y_flat, num_of_class)
            model.num_of_class = num_of_class
            model.weight = np.random.randn(num_of_class, m).astype(np.float32) * 0.01

            y_one_hot = np.zeros((n, num_of_class), dtype=np.float32)
            y_one_hot[np.arange(n), y_flat] = 1.0

            current_lr = float(lr)
            lambda_reg = 0.0005
            batch_size = 256
            lr_decay = 0.98

            progress_start = 25 if label == "TF-IDF" else 60
            progress_end = 60 if label == "TF-IDF" else 95

            for e in range(epoch):
                rand_index = np.random.permutation(n)
                epoch_loss = 0.0

                for start in range(0, n, batch_size):
                    idx = rand_index[start : start + batch_size]
                    Xi = _row_slice(X_b, idx)
                    yi = y_flat[idx]
                    yi_oh = y_one_hot[idx]

                    prob = softmax(_dot(Xi, model.weight.T))
                    sample_weights = class_weights[yi]
                    correct_probs = prob[np.arange(len(idx)), yi]
                    epoch_loss += float(
                        np.sum(sample_weights * -np.log(correct_probs + 1e-15))
                    )

                    grad = (yi_oh - prob) * sample_weights[:, None]
                    grad /= len(idx)

                    if issparse(Xi):
                        weight_update = np.asarray(grad.T.dot(Xi.toarray()))
                    else:
                        weight_update = grad.T.dot(Xi)

                    model.weight += current_lr * weight_update
                    model.weight[:, 1:] *= 1.0 - current_lr * lambda_reg

                current_lr *= lr_decay
                l2 = (lambda_reg / 2.0) * float(np.sum(model.weight[:, 1:] ** 2))
                loss = float(epoch_loss / n) + l2
                loss_list.append(round(loss, 4))

                state["progress"] = progress_start + int(
                    (e + 1) / epoch * (progress_end - progress_start)
                )
                log(
                    f"  [{label}] epoch {e + 1:2d}/{epoch} | loss {loss:.4f} | lr {current_lr:.5f}"
                )

        log(f"Training TF-IDF model ({epoch} epochs)...")
        model1 = SoftmaxRegression()
        patched_fit(model1, X_tr_tfidf, y_tr, tfidf_loss, "TF-IDF")

        log(f"Training N-Gram model ({epoch} epochs)...")
        model2 = SoftmaxRegression()
        patched_fit(model2, X_tr_gram, y_tr_g, ngram_loss, "N-Gram")

        tr_acc1 = model1.score(X_tr_tfidf, y_tr)
        te_acc1 = model1.score(X_te_tfidf, y_te)
        tr_acc2 = model2.score(X_tr_gram, y_tr_g)
        te_acc2 = model2.score(X_te_gram, y_te_g)

        log(f"TF-IDF — train: {tr_acc1:.4f} | test: {te_acc1:.4f}")
        log(f"N-Gram  — train: {tr_acc2:.4f} | test: {te_acc2:.4f}")

        state["model_tfidf"] = model1
        state["model_ngram"] = model2
        state["tfidf"] = tfidf_model
        state["ngram"] = ngram_model
        state["loss_history"] = {"tfidf": tfidf_loss, "ngram": ngram_loss}
        state["train_acc"] = {"tfidf": round(tr_acc1, 4), "ngram": round(tr_acc2, 4)}
        state["test_acc"] = {"tfidf": round(te_acc1, 4), "ngram": round(te_acc2, 4)}
        state["progress"] = 100
        state["status"] = "done"

    except Exception as e:
        import traceback

        state["status"] = "error"
        state["error"] = str(e)
        state["log"].append(f"Error: {e}")
        traceback.print_exc()


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


class TrainRequest(BaseModel):
    n_samples: int = 30000
    epoch: int = 20
    lr: float = 0.3


@app.post("/train")
def start_train(req: TrainRequest):
    if state["status"] == "training":
        return {"ok": False, "msg": "Already training"}
    t = threading.Thread(
        target=run_training, args=(req.n_samples, req.epoch, req.lr), daemon=True
    )
    t.start()
    return {"ok": True}


@app.get("/status")
def get_status():
    return {
        "status": state["status"],
        "progress": state["progress"],
        "log": state["log"][-30:],
        "loss_history": state["loss_history"],
        "train_acc": state["train_acc"],
        "test_acc": state["test_acc"],
        "error": state["error"],
    }


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(req: PredictRequest):
    if state["model_tfidf"] is None:
        return {"error": "Model not trained yet"}

    phrase = [req.text]
    tfidf_feat = csr_matrix(state["tfidf"].transform(phrase, normalize=True))
    gram_feat = csr_matrix(state["ngram"].transform(phrase, normalize=True))

    p1 = int(state["model_tfidf"].predict(tfidf_feat)[0])
    p2 = int(state["model_ngram"].predict(gram_feat)[0])
    proba1 = state["model_tfidf"].predict_proba(tfidf_feat)[0].tolist()
    proba2 = state["model_ngram"].predict_proba(gram_feat)[0].tolist()

    words = clean_text(req.text)
    tfidf_vocab = state["tfidf"].vocab
    tfidf_idf = state["tfidf"].idf
    W = state["model_tfidf"].weight  # shape (3, m+1); column 0 is bias

    word_scores = []
    for w in set(words):
        if w in tfidf_vocab:
            col = tfidf_vocab[w] + 1
            tf = words.count(w) / max(len(words), 1)
            tfidf_val = tf * tfidf_idf.get(w, 0)
            class_score = [float(W[c, col] * tfidf_val) for c in range(3)]
            word_scores.append({"word": w, "scores": class_score})

    word_scores.sort(key=lambda x: max(abs(s) for s in x["scores"]), reverse=True)

    return {
        "tfidf": {"pred": p1, "proba": proba1},
        "ngram": {"pred": p2, "proba": proba2},
        "word_scores": word_scores[:15],
    }
