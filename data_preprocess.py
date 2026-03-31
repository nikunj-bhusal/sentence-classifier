import pandas as pd


def read_data(train_file="dataset/train.tsv"):
    # - `sep="\t"` tells pandas that the columns are separated by tab characters

    # - `train_df["Phrase"]` selects the column named "Phrase" from the table.

    # - `.values` converts that column into a plain array (list-like structure) of values.

    # - Same for the "Sentiment" column.

    # - `return` sends both arrays back to whoever called the function. So the function returns two things : all phrases and all sentiments.
    train_df = pd.read_csv(train_file, sep="\t")
    return train_df["Phrase"].values, train_df["Sentiment"].values


# if __name__ == "__main__":
#     X_data, y_data = read_data()
#     print("train size", len(X_data))
