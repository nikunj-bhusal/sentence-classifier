import pandas as pd


def read_data(train_file="dataset/train.tsv"):
    train_df = pd.read_csv(train_file, sep="\t")
    return train_df["Phrase"].values, train_df["Sentiment"].values


if __name__ == "__main__":
    X_data, y_data = read_data()
    print("train size", len(X_data))
